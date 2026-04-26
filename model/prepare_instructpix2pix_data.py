#!/usr/bin/env python3
"""
Prepare InstructPix2Pix dataset for training.
Downloads from HuggingFace and converts to the format expected by train_uvit.py
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def prepare_from_huggingface(output_dir, num_samples=None, split="train"):
    """Download and prepare InstructPix2Pix dataset from HuggingFace."""
    from datasets import load_dataset
    
    print(f"Loading InstructPix2Pix dataset from HuggingFace...")
    
    if num_samples:
        dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered", split=f"{split}[:{num_samples}]")
        print(f"Loaded {len(dataset)} samples (subset)")
    else:
        dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered", split=split)
        print(f"Loaded {len(dataset)} samples (full dataset)")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mapping file
    mapping = []
    
    print("Processing samples...")
    for idx, sample in enumerate(tqdm(dataset)):
        # Get images and instruction (handle different field names)
        original_image = sample.get('original_image') or sample.get('input_image')
        edited_image = sample.get('edited_image')
        instruction = sample.get('edit_prompt') or sample.get('editing_instruction')
        
        # Save images
        original_path = output_dir / f"source_{idx:06d}.jpg"
        edited_path = output_dir / f"edited_{idx:06d}.jpg"
        
        original_image.save(original_path, quality=95)
        edited_image.save(edited_path, quality=95)
        
        # Add to mapping
        mapping.append({
            "source_image": str(original_path.name),
            "edited_image": str(edited_path.name),
            "editing_instruction": instruction
        })
    
    # Save mapping file
    mapping_path = output_dir / "mapping_file.json"
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n✅ Dataset prepared!")
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
