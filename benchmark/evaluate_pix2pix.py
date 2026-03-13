import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

import open_clip
import lpips
from skimage.metrics import structural_similarity as ssim_metric

def load_rgb(path):
    return Image.open(path).convert("RGB")


def resolve_existing_path(path_str):
    if path_str is None:
        return None
    p = Path(path_str)
    if p.exists():
        return str(p)

    base = p.with_suffix("")
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        alt = str(base) + ext
        if os.path.exists(alt):
            return alt
    return None


def resize_if_needed(img_a, img_b):
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.Resampling.BICUBIC)
    return img_a, img_b


def pil_to_float01(img):
    return np.array(img).astype(np.float32) / 255.0


def crop_to_mask_bbox(img, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    # pad a bit
    pad = 8
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.size[0] - 1, x2 + pad)
    y2 = min(img.size[1] - 1, y2 + pad)
    return img.crop((x1, y1, x2 + 1, y2 + 1))


def get_clip_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def encode_image_text_clip(model, preprocess, tokenizer, image_pil, text, device):
    image = preprocess(image_pil).unsqueeze(0).to(device)
    text = tokenizer([text]).to(device)

    with torch.no_grad():
        image_feat = model.encode_image(image)
        text_feat = model.encode_text(text)

    image_feat = F.normalize(image_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)
    return image_feat, text_feat


def clipscore_from_embeddings(image_feat, text_feat):
    cos = F.cosine_similarity(image_feat, text_feat, dim=-1).item()
    return max(100.0 * cos, 0.0)


def decode_rle_mask(mask_rle, h, w):
    if mask_rle is None:
        return None

    if isinstance(mask_rle, dict):
        counts = mask_rle.get("counts", None)
        size = mask_rle.get("size", None)
        if size is not None and len(size) == 2:
            h, w = int(size[0]), int(size[1])
        mask_rle = counts

    if not isinstance(mask_rle, list):
        return None

    total = h * w
    flat = np.zeros(total, dtype=np.uint8)

    idx = 0
    val = 0
    for run in mask_rle:
        run = int(run)
        if run < 0:
            return None
        end = min(idx + run, total)
        if val == 1:
            flat[idx:end] = 1
        idx = end
        val = 1 - val
        if idx >= total:
            break

    if idx < total and val == 1:
        flat[idx:total] = 1

    mask = flat.reshape((w, h), order="F").T
    return mask.astype(np.float32)


def get_edit_mask_from_ann(ann, img_size):
    w, h = img_size

    if "mask" in ann and ann["mask"] is not None:
        mask = decode_rle_mask(ann["mask"], h, w)
        if mask is not None:
            if mask.shape != (h, w):
                mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)) / 255.0
                mask = (mask > 0.5).astype(np.float32)
            return mask

    return None


def masked_mse(orig_np, edit_np, bg_mask):
    mask3 = bg_mask[..., None]
    diff2 = ((orig_np - edit_np) ** 2) * mask3
    denom = max(mask3.sum() * 3.0, 1e-8)
    return float(diff2.sum() / denom)


def masked_psnr(orig_np, edit_np, bg_mask):
    mse_val = masked_mse(orig_np, edit_np, bg_mask)
    mse_val = max(mse_val, 1e-10)
    return 10.0 * np.log10(1.0 / mse_val)


def masked_ssim(orig_np, edit_np, bg_mask):
    mask3 = bg_mask[..., None]
    orig_bg = orig_np * mask3
    edit_bg = edit_np * mask3
    return float(ssim_metric(orig_bg, edit_bg, channel_axis=2, data_range=1.0))


def masked_lpips(orig_img, edit_img, bg_mask, lpips_model, device):
    orig_np = pil_to_float01(orig_img)
    edit_np = pil_to_float01(edit_img)

    mask3 = bg_mask[..., None]
    orig_bg = orig_np * mask3
    edit_bg = edit_np * mask3

    orig_t = torch.from_numpy(orig_bg).permute(2, 0, 1).unsqueeze(0).to(device)
    edit_t = torch.from_numpy(edit_bg).permute(2, 0, 1).unsqueeze(0).to(device)

    orig_t = orig_t * 2.0 - 1.0
    edit_t = edit_t * 2.0 - 1.0

    with torch.no_grad():
        val = lpips_model(orig_t, edit_t).item()
    return float(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True, help="Path to original PIE-Bench dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to edited (Pix2Pix) images")
    parser.add_argument("--save_csv", type=str, default="pix2pix_metrics.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    mapping_file = os.path.join(args.source_path, "mapping_file.json")
    if not os.path.exists(mapping_file):
        print(f"Error: {mapping_file} not found.")
        return
        
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    print("Loading CLIP...")
    clip_model, clip_preprocess, clip_tokenizer = get_clip_model(device)

    print("Loading LPIPS...")
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    rows = []
    skipped = 0
    no_mask = 0

    for sample_id, ann in tqdm(mapping.items(), total=len(mapping), desc="Evaluating Pix2Pix Results"):
        rel_path = ann["image_path"]
        orig_path = resolve_existing_path(os.path.join(args.source_path, "annotation_images", rel_path))
        edit_path = resolve_existing_path(os.path.join(args.output_path, "annotation_images", rel_path))

        if orig_path is None or edit_path is None:
            skipped += 1
            continue

        try:
            orig_img = load_rgb(orig_path)
            edit_img = load_rgb(edit_path)
            orig_img, edit_img = resize_if_needed(orig_img, edit_img)

            w, h = orig_img.size
            edit_mask = get_edit_mask_from_ann(ann, (w, h))
            if edit_mask is None:
                no_mask += 1
                edit_mask = np.zeros((h, w), dtype=np.float32)

            bg_mask = 1.0 - edit_mask

            orig_np = pil_to_float01(orig_img)
            edit_np = pil_to_float01(edit_img)

            target_prompt = ann["editing_prompt"]

            # Whole CLIPScore
            img_feat_whole, txt_feat = encode_image_text_clip(
                clip_model, clip_preprocess, clip_tokenizer, edit_img, target_prompt, device
            )
            clip_whole = clipscore_from_embeddings(img_feat_whole, txt_feat)

            # Edited-region CLIPScore
            edited_crop = crop_to_mask_bbox(edit_img, edit_mask)
            img_feat_edit, txt_feat_edit = encode_image_text_clip(
                clip_model, clip_preprocess, clip_tokenizer, edited_crop, target_prompt, device
            )
            clip_edited = clipscore_from_embeddings(img_feat_edit, txt_feat_edit)

            mse_raw = masked_mse(orig_np, edit_np, bg_mask)
            psnr_raw = masked_psnr(orig_np, edit_np, bg_mask)
            ssim_raw = masked_ssim(orig_np, edit_np, bg_mask)
            lpips_raw = masked_lpips(orig_img, edit_img, bg_mask, lpips_model, device)

            rows.append({
                "sample_id": sample_id,
                "image_path": rel_path,
                "clip_whole": clip_whole,
                "clip_edited": clip_edited,
                "lpips_raw": lpips_raw,
                "ssim_raw": ssim_raw,
                "psnr_raw": psnr_raw,
                "mse_raw": mse_raw,
                "lpips_x1000": lpips_raw * 1000.0,
                "ssim_x100": ssim_raw * 100.0,
                "mse_x10000": mse_raw * 10000.0,
            })

        except Exception as e:
            print(f"\nSkipping {sample_id} due to error: {e}")
            skipped += 1

    if len(rows) == 0:
        print("No valid samples were evaluated.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.save_csv, index=False)

    print("\n===== PIX2PIX FINAL METRICS =====")
    print(f"Evaluated samples: {len(df)}")
    print(f"Skipped: {skipped}")
    print(f"No-mask fallback: {no_mask}")
    print(f"CLIP Whole:  {df['clip_whole'].mean():.2f}")
    print(f"CLIP Edited: {df['clip_edited'].mean():.2f}")
    print(f"LPIPS:       {df['lpips_x1000'].mean():.2f}")
    print(f"MSE:         {df['mse_x10000'].mean():.2f}")
    print(f"SSIM:        {df['ssim_x100'].mean():.2f}")
    print(f"PSNR:        {df['psnr_raw'].mean():.2f}")
    print(f"\nSaved per-image metrics to: {args.save_csv}")


if __name__ == "__main__":
    main()
