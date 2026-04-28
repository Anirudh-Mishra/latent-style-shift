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
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, LCMScheduler, DDPMScheduler, UNet2DConditionModel
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


class EncodedLatentDataset(Dataset):
    """Fast dataset that loads pre-encoded VAE latents + CLIP embeddings from disk.

    Create with prepare_encoded_dataset.py. Eliminates VAE and text-encoder
    inference from every training step (~70 ms/step savings on a 5090).
    """
    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        self.src = torch.load(data_dir / "source_latents.pt", weights_only=True)
        self.tgt = torch.load(data_dir / "target_latents.pt", weights_only=True)
        self.txt = torch.load(data_dir / "text_embeddings.pt", weights_only=True)
        assert len(self.src) == len(self.tgt) == len(self.txt), "Size mismatch in encoded dataset"
        print(f"Loaded encoded dataset: {len(self.src)} samples from {data_dir}")

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        # Return float32 so the training loop doesn't need special dtype handling
        return self.src[idx].float(), self.tgt[idx].float(), self.txt[idx].float()


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
    # BF16 is natively fast on Blackwell (5090) and doesn't need a GradScaler
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    print(f"Device: {device}")
    print(f"U-ViT size: {args.uvit_size}")
    print(f"AMP dtype: {amp_dtype}")

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

    # Frozen teacher UNet for knowledge distillation.
    # When --distill is set the UViT learns to match UNet predictions (which
    # already embed strong text conditioning) rather than the raw noise vector.
    teacher_unet = None
    if args.distill:
        print("Loading frozen UNet teacher for knowledge distillation...")
        teacher_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
        teacher_unet = teacher_unet.to(device)
        teacher_unet.eval()
        teacher_unet.requires_grad_(False)
        print("UNet teacher loaded and frozen.")

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
            # torch.compile saves keys prefixed with "_orig_mod." — strip it
            if any(k.startswith("_orig_mod.") for k in state_dict):
                state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

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

    if not args.encoded_data_dir and not args.data_dir:
        raise ValueError("Provide --data_dir (raw images) or --encoded_data_dir (pre-encoded latents).")

    # Compute null text embedding (empty string) before text encoder is offloaded.
    # The source stream during inference uses null/empty text, so training the
    # source stream with null text closes the training-inference gap.
    with torch.no_grad():
        null_ids = tokenizer(
            "", padding="max_length", max_length=77, return_tensors="pt"
        ).input_ids.to(device)
        null_text_emb = text_encoder(null_ids)[0].float().cpu()  # [1, 77, 768]
    print("Computed null text embedding for source stream conditioning.")

    if args.encoded_data_dir:
        print(f"Using pre-encoded dataset from {args.encoded_data_dir}")
        dataset = EncodedLatentDataset(args.encoded_data_dir)
        # Encoded dataset returns (src_latent, tgt_latent, text_embedding) — no
        # VAE or CLIP needed during training, so unload them to free VRAM.
        vae = vae.cpu()
        text_encoder = text_encoder.cpu()
        torch.cuda.empty_cache()
    else:
        dataset = ImageTextDataset(args.data_dir, image_size=args.image_size, tokenizer=tokenizer)

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

    # torch.compile speeds up the UViT forward+backward ~20-40% on Blackwell.
    # Compiles once on the first step (~2 min) then runs optimised kernels.
    if args.compile:
        print("Compiling UViT with torch.compile (max-autotune)...")
        model = torch.compile(model, mode="max-autotune")

    optimizer = torch.optim.AdamW(
        model.parameters() if not args.compile else model._orig_mod.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # BF16 is numerically stable and doesn't need a GradScaler.
    # FP16 keeps the scaler for safety.
    scaler = None
    if args.use_amp and not args.bf16:
        scaler = torch.cuda.amp.GradScaler()

    total_steps = len(dataloader) * args.num_epochs
    warmup_steps = min(args.warmup_steps, total_steps)
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
            raw_sd = ckpt.get("model", {})
            if any(k.startswith("_orig_mod.") for k in raw_sd):
                raw_sd = {k.replace("_orig_mod.", "", 1): v for k, v in raw_sd.items()}
            # Load into the uncompiled backbone — compiled OptimizedModule uses
            # _orig_mod. prefix so stripped keys would all silently mismatch with strict=False.
            target_module = model._orig_mod if hasattr(model, '_orig_mod') else model
            target_module.load_state_dict(raw_sd, strict=False)
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
            if args.reset_epoch:
                global_step = 0
                start_epoch = 0
                best_loss = float("inf")
                print("Epoch/step counters reset (cross-stage load)")
            else:
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

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            leave=True,
            dynamic_ncols=True,
        )

        class StoredAttnInjector:
            """Injects source cross-attention maps into the target stream.

            Uses soft blending (alpha < 1.0) during training so gradients still
            flow through the (1-alpha) computed portion back to to_q and to_k.
            Hard injection (alpha=1.0) is used at inference only.
            """
            def __init__(self, store, alpha=0.8):
                self.store = {k: [t.detach() for t in v] for k, v in store.items()}
                self.ptrs = {k: 0 for k in self.store}
                self.alpha = alpha

            def __call__(self, attn, is_cross: bool, place_in_unet: str):
                if not is_cross:
                    return attn
                key = f"{place_in_unet}_cross"
                lst = self.store.get(key, [])
                if len(lst) == 0:
                    return attn
                idx = self.ptrs.get(key, 0)
                stored = lst[min(idx, len(lst) - 1)].to(attn.device)
                self.ptrs[key] = min(idx + 1, len(lst) - 1)
                # Soft blend: gradients flow through (1-alpha)*attn to to_q and to_k
                return self.alpha * stored + (1.0 - self.alpha) * attn

            def self_attn_forward(self, q, k, v, h):
                return q, k, v

        for batch_idx, batch in enumerate(progress_bar):
            if args.encoded_data_dir:
                # Fast path: pre-encoded latents + embeddings, no VAE/CLIP needed
                source_latents, target_latents, encoder_hidden_states = batch
                source_latents = source_latents.to(device)
                target_latents = target_latents.to(device)
                encoder_hidden_states = encoder_hidden_states.to(device)
            else:
                src_images, edt_images, input_ids = batch
                src_images = src_images.to(device, dtype=dtype)
                edt_images = edt_images.to(device, dtype=dtype)
                input_ids = input_ids.to(device)
                with torch.no_grad():
                    source_latents = vae.encode(src_images).latent_dist.sample() * vae.config.scaling_factor
                    target_latents = vae.encode(edt_images).latent_dist.sample() * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(input_ids)[0]

            B = source_latents.shape[0]
            # Null text for source stream — matches inference where source stream
            # uses empty/null conditioning, not the editing instruction.
            null_emb = null_text_emb.expand(B, -1, -1).to(device)

            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (B,), device=device, dtype=torch.long,
            )

            # Noise both source and target latents independently
            source_noise = torch.randn_like(source_latents)
            target_noise = torch.randn_like(target_latents)
            noisy_source = scheduler.add_noise(source_latents, source_noise, timesteps)
            noisy_target = scheduler.add_noise(target_latents, target_noise, timesteps)

            # source_conditioned mode concatenates clean source latent as extra channels.
            # Pass it to all adapter calls so the backbone sees it instead of zeros.
            src_kwargs = {"source_latent": source_latents} if args.source_conditioned else {}

            # --- Teacher targets (distillation) or raw noise (fine-tune) ---
            if teacher_unet is not None:
                with torch.no_grad():
                    # Source stream: null text — matches inference source stream
                    source_teacher = teacher_unet(
                        noisy_source, timesteps,
                        encoder_hidden_states=null_emb,
                    ).sample.detach()
                    # Target stream: editing instruction text
                    target_teacher = teacher_unet(
                        noisy_target, timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample.detach()
            else:
                source_teacher = source_noise
                target_teacher = target_noise

            # --- Capture source attention maps (null text + noisy source) ---
            # Use null text and noisy source to match the actual inference source stream.
            attention_store = ptp_utils.AttentionStore()
            uvit_register(adapter, attention_store)
            attention_store.reset()
            with torch.no_grad():
                _ = adapter(noisy_source, timesteps, null_emb, **src_kwargs).sample
            stored_maps = attention_store.attention_store if len(attention_store.attention_store) > 0 else attention_store.step_store
            uvit_unregister(adapter)

            # --- Source denoising loss (null text, no attention injection) ---
            # Must run BEFORE injector is registered — source stream is standalone.
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.use_amp):
                source_loss = F.mse_loss(
                    adapter(noisy_source, timesteps, null_emb, **src_kwargs).sample,
                    source_teacher,
                )

            # --- Target denoising loss (editing text, source attention injected) ---
            injector = StoredAttnInjector(stored_maps)
            uvit_register(adapter, injector)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.use_amp):
                target_loss = F.mse_loss(
                    adapter(noisy_target, timesteps, encoder_hidden_states, **src_kwargs).sample,
                    target_teacher,
                )
            uvit_unregister(adapter)

            loss = 0.5 * source_loss + 0.5 * target_loss

            if args.use_amp and scaler is not None:
                loss_value = loss.item()
                (loss / args.grad_accum_steps).backward()
                if (batch_idx + 1) % args.grad_accum_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    lr_scheduler.step()
            else:
                loss_value = loss.item()
                (loss / args.grad_accum_steps).backward()
                if (batch_idx + 1) % args.grad_accum_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

            epoch_loss += loss_value
            global_step += 1

            avg = epoch_loss / (batch_idx + 1)
            lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(loss=f"{loss_value:.4f}", avg=f"{avg:.4f}", lr=f"{lr:.2e}")

            if global_step % args.log_every == 0:
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

    parser.add_argument("--data_dir", type=str, default=None,
                        help="Raw image dataset directory. Not required when --encoded_data_dir is provided.")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=64)

    parser.add_argument("--uvit_size", type=str, default="mid",
                        choices=["small", "mid", "large"])
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--source_conditioned", action="store_true",
                        help="Concatenate clean source latent as extra input channels (doubles patch embed in_chans to 8). "
                             "Must match the flag used when the model was created.")
    parser.add_argument("--distill", action="store_true",
                        help="Knowledge distillation: use frozen LCM UNet predictions as training targets "
                             "instead of raw noise. Strongly recommended when starting from MAE init.")
    parser.add_argument("--encoded_data_dir", type=str, default=None,
                        help="Directory of pre-encoded latents+embeddings from prepare_encoded_dataset.py. "
                             "Skips VAE/CLIP inference every step (~70 ms/step savings).")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model with torch.compile(max-autotune) for ~20-40%% faster kernels on Blackwell.")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 autocast instead of FP16. Recommended on 5090/H100 — no GradScaler needed.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset_epoch", action="store_true",
                        help="Reset epoch/step counters when loading a checkpoint from a previous stage.")

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
