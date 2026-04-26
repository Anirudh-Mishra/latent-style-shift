#!/usr/bin/env python3
"""
Prepare InstructPix2Pix dataset for training (robust version).
Downloads from HuggingFace and converts to the format expected by train_uvit.py

This script is more tolerant to schema changes in HF dataset entries.
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def prepare_from_huggingface(output_dir, num_samples=None, split="train"):
    """Stream and prepare InstructPix2Pix dataset from HuggingFace."""
    from datasets import load_dataset

    print("Streaming InstructPix2Pix dataset from HuggingFace...")

    dataset = load_dataset(
        "timbrooks/instructpix2pix-clip-filtered",
        split=split,
        streaming=True
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = []

    max_samples = num_samples if num_samples is not None else float("inf")

    print("Processing samples...")
    first = True
    for idx, sample in enumerate(tqdm(dataset, total=num_samples)):
        if idx >= max_samples:
            break

        # On first sample, show available keys to help debug schema mismatches
        if first:
            try:
                print("Sample keys:", list(sample.keys()))
            except Exception:
                print("Sample appears not a mapping; skipping key print")
            first = False

        # Helper: try a list of possible field names
        def pick_field(d, candidates):
            for k in candidates:
                if k in d:
                    return d[k]
            return None

        original_image = pick_field(sample, ["input_image", "original_image", "image", "source_image", "source"]) 
        edited_image = pick_field(sample, ["edited_image", "target_image", "edited", "output_image"]) 
        instruction = pick_field(sample, ["edit_prompt", "instruction", "prompt", "caption", "text"]) 

        if original_image is None or edited_image is None:
            print(f"Skipping sample {idx}: missing image fields (keys: {list(sample.keys())})")
            continue

        # Convert various image representations to PIL.Image
        from io import BytesIO
        def to_pil(img):
            # pass through if already PIL
            try:
                from PIL import Image as PilImage
                if isinstance(img, PilImage.Image):
                    return img
            except Exception:
                pass
            # if dict with 'image' inside
            if isinstance(img, dict) and 'image' in img:
                return to_pil(img['image'])
            # if bytes
            if isinstance(img, (bytes, bytearray)):
                return Image.open(BytesIO(img)).convert('RGB')
            # if numpy array
            try:
                import numpy as _np
                if isinstance(img, _np.ndarray):
                    return Image.fromarray(img)
            except Exception:
                pass
            # if it's a list, take first
            if isinstance(img, (list, tuple)) and len(img) > 0:
                return to_pil(img[0])
            # fallback: try to open as path
            try:
                return Image.open(str(img)).convert('RGB')
            except Exception:
                raise RuntimeError(f"Unable to convert image field to PIL for sample idx={idx}")

        original_image = to_pil(original_image)
        edited_image = to_pil(edited_image)

        original_path = output_dir / f"source_{idx:06d}.jpg"
        edited_path = output_dir / f"edited_{idx:06d}.jpg"

        original_image.save(original_path, quality=95)
        edited_image.save(edited_path, quality=95)

        mapping.append({
            "source_image": original_path.name,
            "edited_image": edited_path.name,
            "editing_instruction": instruction if instruction is not None else ""
        })

    mapping_path = output_dir / "mapping_file.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print("\n✅ Dataset prepared!")
    print(f"   Samples: {len(mapping)}")
    print(f"   Images: {len(mapping) * 2}")
    print(f"   Location: {output_dir}")
    print(f"   Mapping: {mapping_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for prepared dataset')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to download (None = all 313k)')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use')
    args = parser.parse_args()
    
    # Check if datasets library is installed
    try:
        import datasets
    except ImportError:
        print("ERROR: 'datasets' library not installed")
        print("Install with: pip install datasets")
        return
    
    prepare_from_huggingface(args.output_dir, args.num_samples, args.split)


if __name__ == '__main__':
    main()
