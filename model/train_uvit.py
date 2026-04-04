import os
import sys
import math
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import AutoencoderKL, LCMScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer

from uvit_backbone import UViTBackbone, UVIT_CONFIGS


class ImageTextDataset(Dataset):
    def __init__(self, data_dir, image_size=512, tokenizer=None, max_length=77):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []

        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            if isinstance(metadata, list):
                for entry in metadata:
                    img_path = self.data_dir / entry["file_name"]
                    if img_path.exists():
                        self.pairs.append((str(img_path), entry["text"]))
            elif isinstance(metadata, dict):
                for fname, text in metadata.items():
                    img_path = self.data_dir / fname
                    if img_path.exists():
                        self.pairs.append((str(img_path), text))
        else:
            for img_path in sorted(self.data_dir.glob("*")):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    txt_path = img_path.with_suffix(".txt")
                    if txt_path.exists():
                        caption = txt_path.read_text().strip()
                    else:
                        caption = ""
                    self.pairs.append((str(img_path), caption))

        if len(self.pairs) == 0:
            raise ValueError(f"No image-text pairs found in {data_dir}")
        print(f"Found {len(self.pairs)} image-text pairs")

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)

        return image, input_ids


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Device: {device}")
    print(f"U-ViT size: {args.uvit_size}")

    model_id = "SimianLuo/LCM_Dreamshaper_v7"
    
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    vae.requires_grad_(False)

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    model = UViTBackbone.from_preset(
        args.uvit_size,
        img_size=args.latent_size,
        patch_size=args.patch_size,
        in_chans=4,
        context_dim=768,
    )
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        state_dict = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device, dtype=dtype)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-ViT parameters: {num_params / 1e6:.1f}M")

    scheduler = LCMScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler.set_timesteps(1000, device=device)

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
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    total_steps = len(dataloader) * args.num_epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nStarting training for {args.num_epochs} epochs ({total_steps} steps)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {args.output_dir}\n")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, input_ids) in enumerate(dataloader):
            images = images.to(device, dtype=dtype)
            input_ids = input_ids.to(device)

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device, dtype=torch.long,
            )

            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            noise_pred = model(noisy_latents, timesteps, encoder_hidden_states)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                avg = epoch_loss / (batch_idx + 1)
                lr = optimizer.param_groups[0]["lr"]
                print(f"  [Step {global_step}] loss={loss.item():.4f}  avg={avg:.4f}  lr={lr:.2e}")

        epoch_loss /= len(dataloader)
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}  loss={epoch_loss:.4f}  time={elapsed:.1f}s")

        if (epoch + 1) % args.save_every == 0 or epoch_loss < best_loss:
            ckpt_path = os.path.join(args.output_dir, f"uvit_{args.uvit_size}_epoch{epoch + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = os.path.join(args.output_dir, f"uvit_{args.uvit_size}_best.pt")
                torch.save(model.state_dict(), best_path)
                print(f"New best model: {best_path}")

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
    parser.add_argument("--resume", type=str, default=None)

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

    args = parser.parse_args()
    train(args)
