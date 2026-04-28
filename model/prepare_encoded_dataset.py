"""
Pre-encode the InstructPix2Pix dataset to VAE latents + CLIP embeddings.

Run this ONCE before training. The output directory can then be passed to
train_uvit.py via --encoded_data_dir to skip VAE/CLIP inference every step.

Estimated time on a 5090: ~2 minutes for 50k pairs.
Disk usage: ~7 GB (latents fp16 + embeddings fp16).

Usage:
    python prepare_encoded_dataset.py \
        --data_dir /home/avid/dl_data/instructpix2pix_50k/ \
        --out_dir  /home/avid/dl_data/instructpix2pix_50k_encoded/ \
        --batch_size 64
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer


class RawPairDataset(Dataset):
    """Minimal dataset that just returns (src_image, edt_image, input_ids)."""
    def __init__(self, data_dir, image_size, tokenizer, max_length=77):
        self.pairs = []
        data_dir = Path(data_dir)

        meta = data_dir / "mapping_file.json"
        if not meta.exists():
            raise FileNotFoundError(f"mapping_file.json not found in {data_dir}")

        with open(meta) as f:
            metadata = json.load(f)

        def _find(base, entry, keys):
            for k in keys:
                if k in entry and entry[k]:
                    p = Path(entry[k])
                    if not p.is_absolute():
                        p = base / entry[k]
                    if p.exists():
                        return str(p)
            return None

        items = metadata if isinstance(metadata, list) else list(metadata.values())
        for entry in items:
            if not isinstance(entry, dict):
                continue
            src = _find(data_dir, entry, ["source_image", "source", "original_image", "original_path"])
            edt = _find(data_dir, entry, ["edited_image", "edited", "edited_image_path", "image_path"])
            text = entry.get("editing_instruction", entry.get("editing_prompt", entry.get("text", "")))
            if src and edt:
                self.pairs.append((src, edt, text))

        if not self.pairs:
            raise ValueError("No source/edited pairs found")
        print(f"Found {len(self.pairs)} pairs")

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_path, edt_path, caption = self.pairs[idx]
        src = self.transform(Image.open(src_path).convert("RGB"))
        edt = self.transform(Image.open(edt_path).convert("RGB"))
        ids = self.tokenizer(
            caption, padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.squeeze(0)
        return src, edt, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_id", default="SimianLuo/LCM_Dreamshaper_v7")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if (out / "source_latents.pt").exists() and \
       (out / "target_latents.pt").exists() and \
       (out / "text_embeddings.pt").exists():
        print(f"Encoded dataset already exists at {out}. Delete to re-encode.")
        return

    print("Loading VAE and text encoder...")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae",
                                        torch_dtype=dtype).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder",
                                                 torch_dtype=dtype).to(device).eval()
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    dataset = RawPairDataset(args.data_dir, args.image_size, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, pin_memory=True, drop_last=False)

    N = len(dataset)
    # Pre-allocate output tensors in CPU memory
    src_lat = torch.zeros(N, 4, args.image_size // 8, args.image_size // 8, dtype=torch.float16)
    tgt_lat = torch.zeros_like(src_lat)
    txt_emb = torch.zeros(N, 77, text_encoder.config.hidden_size, dtype=torch.float16)

    idx = 0
    with torch.no_grad():
        for src_imgs, edt_imgs, input_ids in tqdm(loader, desc="Encoding"):
            B = src_imgs.shape[0]
            src_imgs = src_imgs.to(device, dtype=dtype)
            edt_imgs = edt_imgs.to(device, dtype=dtype)
            input_ids = input_ids.to(device)

            sl = vae.encode(src_imgs).latent_dist.sample() * vae.config.scaling_factor
            tl = vae.encode(edt_imgs).latent_dist.sample() * vae.config.scaling_factor
            te = text_encoder(input_ids)[0]

            src_lat[idx:idx+B] = sl.cpu().to(torch.float16)
            tgt_lat[idx:idx+B] = tl.cpu().to(torch.float16)
            txt_emb[idx:idx+B] = te.cpu().to(torch.float16)
            idx += B

    print(f"Saving to {out} ...")
    torch.save(src_lat, out / "source_latents.pt")
    torch.save(tgt_lat, out / "target_latents.pt")
    torch.save(txt_emb, out / "text_embeddings.pt")

    src_gb = src_lat.nbytes / 1e9
    tgt_gb = tgt_lat.nbytes / 1e9
    txt_gb = txt_emb.nbytes / 1e9
    print(f"Done. Disk usage: {src_gb:.2f}+{tgt_gb:.2f}+{txt_gb:.2f} = {src_gb+tgt_gb+txt_gb:.2f} GB")
    print(f"Pass --encoded_data_dir {out} to train_uvit.py to use this cache.")


if __name__ == "__main__":
    main()
