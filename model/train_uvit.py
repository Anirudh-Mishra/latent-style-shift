import os
import sys
import math
import json
import argparse
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import AutoencoderKL, LCMScheduler, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer

from uvit_backbone import UViTBackbone, UVIT_CONFIGS


def prepare_coco_mapping(coco_json_path: str, images_dir: str, out_dir: str):
    """Create a mapping_file.json for train_uvit from COCO captions JSON.

    mapping entries will contain absolute paths to images (no copying).
    """
    coco_json_path = Path(coco_json_path)
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    if not coco_json_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {coco_json_path}")
    with open(coco_json_path) as f:
        coco = json.load(f)

    # build id -> filename
    id2fname = {img["id"]: img["file_name"] for img in coco.get("images", [])}
    # choose one caption per image (first encountered)
    mapping = {}
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        fname = id2fname.get(img_id)
        if fname is None:
            continue
        if fname not in mapping:
            img_path = images_dir / fname
            mapping[fname] = {
                "image_path": str(img_path.resolve()),
                "editing_instruction": ann.get("caption", ""),
            }

    out_path = out_dir / "mapping_file.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Wrote COCO mapping to {out_path} with {len(mapping)} entries")


class ImageTextDataset(Dataset):
    def __init__(self, data_dir, image_size=512, tokenizer=None, max_length=77):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []

        metadata_path = self.data_dir / "mapping_file.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            # Expect mapping entries to provide both a source and edited image path
            def _first_existing_path(base: Path, entry: dict, keys: list):
                for k in keys:
                    if k in entry and entry[k]:
                        p = Path(entry[k])
                        if not p.is_absolute():
                            p = base / entry[k]
                        if p.exists():
                            return str(p)
                return None

            if isinstance(metadata, list):
                for entry in metadata:
                    # Entry must be a dict with source & edited image paths and text
                    if not isinstance(entry, dict):
                        continue
                    src = _first_existing_path(self.data_dir, entry, ["source_image", "source", "original_image", "original_path"])
                    edt = _first_existing_path(self.data_dir, entry, ["edited_image", "edited", "edited_image_path", "image_path"])
                    text = entry.get("text", entry.get("editing_instruction", entry.get("editing_prompt", "")))
                    if src is None or edt is None:
                        # skip entries that don't contain both paths
                        continue
                    self.pairs.append((src, edt, text))
            elif isinstance(metadata, dict):
                for fname, dict_data in metadata.items():
                    if not isinstance(dict_data, dict):
                        continue
                    src = _first_existing_path(self.data_dir, dict_data, ["source_image", "source", "original_image", "original_path"])
                    edt = _first_existing_path(self.data_dir, dict_data, ["edited_image", "edited", "edited_image_path", "image_path"])
                    text = dict_data.get("editing_instruction", dict_data.get("editing_prompt", dict_data.get("text", "")))
                    if src is None or edt is None:
                        # If only a single image is present, skip — training requires pairs
                        continue
                    self.pairs.append((src, edt, text))
        else:
            # Fallback: look for image pairs in the directory
            # Expected naming: source_xxx.jpg and edited_xxx.jpg
            img_files = sorted(self.data_dir.glob("*"))
            source_files = [f for f in img_files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and "source" in f.stem.lower()]
            
            for src_path in source_files:
                # Try to find corresponding edited image
                base_name = src_path.stem.replace("source_", "").replace("source", "")
                edited_candidates = [
                    src_path.parent / f"edited_{base_name}{src_path.suffix}",
                    src_path.parent / f"edited{base_name}{src_path.suffix}",
                    src_path.parent / f"{base_name}_edited{src_path.suffix}",
                    src_path.parent / f"{base_name}edited{src_path.suffix}",
                ]
                
                edt_path = None
                for candidate in edited_candidates:
                    if candidate.exists():
                        edt_path = candidate
                        break
                
                if edt_path is None:
                    continue
                
                # Look for caption file
                txt_path = src_path.with_suffix(".txt")
                if txt_path.exists():
                    caption = txt_path.read_text().strip()
                else:
                    # Try edited image txt
                    txt_path = edt_path.with_suffix(".txt")
                    caption = txt_path.read_text().strip() if txt_path.exists() else ""
                
                self.pairs.append((str(src_path), str(edt_path), caption))

        if len(self.pairs) == 0:
            raise ValueError(f"No source/edited image pairs found in {self.data_dir}; mapping_file.json must contain both source and edited image paths per entry")
        print(f"Found {len(self.pairs)} image-text pairs")

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_path, edt_path, caption = self.pairs[idx]
        src_image = Image.open(src_path).convert("RGB")
        edt_image = Image.open(edt_path).convert("RGB")
        src_image = self.transform(src_image)
        edt_image = self.transform(edt_image)

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)

        return src_image, edt_image, input_ids


def train(args):
    # deterministic seeding
    def set_seed(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Device: {device}")
    print(f"U-ViT size: {args.uvit_size}")

    model_id = "SimianLuo/LCM_Dreamshaper_v7"
    
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    if args.freeze_vae:
        vae.requires_grad_(False)

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    if args.freeze_text_encoder:
        text_encoder.requires_grad_(False)

    # If resuming, detect model dimensions from checkpoint first
    model_overrides = {
        'img_size': args.latent_size,
        'patch_size': args.patch_size,
        'in_chans': 4,
        'context_dim': 768,
        'source_conditioned': args.source_conditioned,
    }
    
    if args.resume:
        resume_path = args.resume
        print(f"Will resume from {resume_path} if checkpoint dict contains optimizer/state info")
        
        # Load checkpoint to detect dimensions
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
            
            # Detect embed_dim
            if 'in_blocks.0.norm1.weight' in state_dict:
                embed_dim = state_dict['in_blocks.0.norm1.weight'].shape[0]
                print(f"Detected embed_dim={embed_dim} from checkpoint")
                model_overrides['embed_dim'] = embed_dim
            
            # Detect depth
            num_in = sum(1 for k in state_dict.keys() if k.startswith('in_blocks.') and '.norm1.weight' in k)
            num_out = sum(1 for k in state_dict.keys() if k.startswith('out_blocks.') and '.norm1.weight' in k)
            depth = num_in + 1 + num_out
            print(f"Detected depth={depth} from checkpoint")
            model_overrides['depth'] = depth
    
    model = UViTBackbone.from_preset(
        args.uvit_size,
        **model_overrides
    )
    
    model = model.to(device, dtype=dtype)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-ViT parameters: {num_params / 1e6:.1f}M")

    # Wrap backbone in adapter so we can register UAC processors during training
    from uvit_adapter import UViTAdapter, register_attention_control as uvit_register, unregister_attention_control as uvit_unregister
    import ptp_utils
    adapter = UViTAdapter(model)
    adapter = adapter.to(device, dtype=dtype)

    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    dataset = ImageTextDataset(
        args.data_dir,
        image_size=args.image_size,
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=(torch.Generator().manual_seed(args.seed) if args.seed is not None else None),
        worker_init_fn=(
            (lambda wid: torch.manual_seed(args.seed + wid)) if args.seed is not None else None
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # mixed precision scaler
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    total_steps = len(dataloader) * args.num_epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    print(f"  Effective warmup steps: {warmup_steps} (requested {args.warmup_steps})")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume support: load optimizer, lr_scheduler, scaler, and training state if available
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    if args.resume:
        ckpt_path = resume_path
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt.get("model", {}), strict=False)
            opt_state = ckpt.get("optimizer", None)
            if opt_state is not None:
                try:
                    optimizer.load_state_dict(opt_state)
                except Exception:
                    print("Warning: failed to fully load optimizer state; continuing with fresh optimizer")
            lr_state = ckpt.get("lr_scheduler", None)
            if lr_state is not None:
                try:
                    lr_scheduler.load_state_dict(lr_state)
                except Exception:
                    print("Warning: failed to load lr_scheduler state")
            if args.use_amp and "scaler" in ckpt and ckpt.get("scaler") is not None:
                try:
                    scaler.load_state_dict(ckpt.get("scaler"))
                except Exception:
                    print("Warning: failed to load AMP scaler state")
            global_step = ckpt.get("global_step", 0)
            start_epoch = ckpt.get("epoch", 0)
            best_loss = ckpt.get("best_loss", best_loss)
            print(f"Resuming from epoch {start_epoch}, global_step {global_step}")
        else:
            print(f"Resume path {ckpt_path} not found; starting fresh training")

    print(f"\nStarting training for {args.num_epochs} epochs ({total_steps} steps)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {args.output_dir}\n")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        # Clear any stale gradients left by a partial accumulation window at the
        # end of the previous epoch so they don't bleed into this epoch's first step.
        optimizer.zero_grad()

        for batch_idx, (src_images, edt_images, input_ids) in enumerate(dataloader):
            src_images = src_images.to(device, dtype=dtype)
            edt_images = edt_images.to(device, dtype=dtype)
            input_ids = input_ids.to(device)

            # Encode source (clean) and edited (to-be-noised) images with VAE
            with torch.no_grad():
                source_latents = vae.encode(src_images).latent_dist.sample()
                source_latents = source_latents * vae.config.scaling_factor

                target_latents = vae.encode(edt_images).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (target_latents.shape[0],), device=device, dtype=torch.long,
            )

            # Noise only the target (edited) latents — the model predicts this noise
            noise = torch.randn_like(target_latents)
            noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)

            # --- Capture source attention maps using AttentionStore ---
            attention_store = ptp_utils.AttentionStore()
            uvit_register(adapter, attention_store)
            attention_store.reset()
            with torch.no_grad():
                _ = adapter(source_latents, timesteps, encoder_hidden_states, source_latent=source_latents if args.source_conditioned else None).sample

            stored_maps = attention_store.attention_store if len(attention_store.attention_store) > 0 else attention_store.step_store
            uvit_unregister(adapter)

            class StoredAttnInjector:
                def __init__(self, store):
                    self.store = {k: [t.detach() for t in v] for k, v in store.items()}
                    self.ptrs = {k: 0 for k in self.store}

                def __call__(self, attn, is_cross: bool, place_in_unet: str):
                    if not is_cross:
                        return attn
                    key = f"{place_in_unet}_cross"
                    lst = self.store.get(key, [])
                    if len(lst) == 0:
                        return attn
                    idx = self.ptrs.get(key, 0)
                    out = lst[min(idx, len(lst) - 1)].to(attn.device)
                    self.ptrs[key] = min(idx + 1, len(lst) - 1)
                    return out

                def self_attn_forward(self, q, k, v, h):
                    return q, k, v

            injector = StoredAttnInjector(stored_maps)
            uvit_register(adapter, injector)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    noise_pred = adapter(noisy_latents, timesteps, encoder_hidden_states, source_latent=source_latents if args.source_conditioned else None).sample
                    loss = F.mse_loss(noise_pred, noise)
                loss_value = loss.item()  # record true loss before division
                loss = loss / args.grad_accum_steps
                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.grad_accum_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    lr_scheduler.step()
            else:
                noise_pred = adapter(noisy_latents, timesteps, encoder_hidden_states, source_latent=source_latents if args.source_conditioned else None).sample
                loss = F.mse_loss(noise_pred, noise)
                loss_value = loss.item()  # record true loss before division
                loss = loss / args.grad_accum_steps
                loss.backward()
                if (batch_idx + 1) % args.grad_accum_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

            # Unregister injector after the forward so subsequent batches start fresh
            try:
                uvit_unregister(adapter)
            except Exception:
                pass

            epoch_loss += loss_value
            global_step += 1

            if global_step % args.log_every == 0:
                avg = epoch_loss / (batch_idx + 1)
                lr = optimizer.param_groups[0]["lr"]
                print(f"  [Step {global_step}] loss={loss_value:.4f}  avg={avg:.4f}  lr={lr:.2e}")

        epoch_loss /= len(dataloader)
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}  loss={epoch_loss:.4f}  time={elapsed:.1f}s")

        # Save full checkpoint (model + optimizer + scheduler + scaler + metadata)
        if (epoch + 1) % args.save_every == 0 or epoch_loss < best_loss:
            ckpt_path = os.path.join(args.output_dir, f"uvit_{args.uvit_size}_epoch{epoch + 1}.pt")
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "global_step": global_step,
                "epoch": epoch + 1,
                "best_loss": best_loss,
            }
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(args.output_dir, f"uvit_{args.uvit_size}_best.pt")
            best_ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "global_step": global_step,
                "epoch": epoch + 1,
                "best_loss": best_loss,
            }
            torch.save(best_ckpt, best_path)
            print(f"New best model: {best_path}")

        # always save metadata for this epoch
        meta = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "epoch_loss": epoch_loss,
            "best_loss": best_loss,
            "args": vars(args),
        }
        try:
            with open(os.path.join(args.output_dir, f"uvit_{args.uvit_size}_epoch{epoch + 1}.meta.json"), "w") as mf:
                json.dump(meta, mf, indent=2)
        except Exception:
            pass

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=64)

    parser.add_argument("--uvit_size", type=str, default="mid",
                        choices=["small", "mid", "large"])
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--source_conditioned", action="store_true",
                        help="Concatenate clean source latent as extra input channels (doubles patch embed in_chans to 8). "
                             "Must match the flag used when the model was created.")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of steps to accumulate gradients before optimizer step")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision (AMP)")
    parser.add_argument("--freeze_vae", dest="freeze_vae", action="store_true", help="Freeze pretrained VAE")
    parser.add_argument("--no_freeze_vae", dest="freeze_vae", action="store_false", help="Do not freeze VAE")
    parser.add_argument("--freeze_text_encoder", dest="freeze_text_encoder", action="store_true", help="Freeze CLIP text encoder")
    parser.add_argument("--no_freeze_text_encoder", dest="freeze_text_encoder", action="store_false", help="Do not freeze text encoder")
    parser.set_defaults(freeze_vae=True, freeze_text_encoder=True)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--output_dir", type=str, default="./uvit_checkpoints")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--coco_annotations", type=str, default=None,
                        help="Path to COCO captions JSON; when set, a mapping_file.json will be generated in --data_dir")
    parser.add_argument("--coco_images_dir", type=str, default=None,
                        help="Path to COCO images (train2017). If not set, will look in parent folder of annotations for 'train2017'.")

    args = parser.parse_args()
    # If provided, prepare mapping from COCO annotations
    if args.coco_annotations is not None:
        images_dir = args.coco_images_dir
        if images_dir is None:
            # try to infer
            annp = Path(args.coco_annotations)
            cand = annp.parent.parent / "train2017"
            if cand.exists():
                images_dir = str(cand)
            else:
                raise ValueError("--coco_images_dir must be provided or train2017 must be next to annotations")
        prepare_coco_mapping(args.coco_annotations, images_dir, args.data_dir)

    train(args)
