#!/usr/bin/env python3
"""
Prepare PIE-Bench dataset for training by creating source/edited image pairs.
PIE-Bench only has edited images, so we need to generate source images.
"""

import json
import argparse
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm


def prepare_pie_bench(source_path, output_path):
    """
    Prepare PIE-Bench for training.
    
    Note: PIE-Bench only contains edited images, not source images.
    For training, we need source/edited pairs.
    
    Options:
    1. Use the edited images as both source and target (identity mapping)
    2. Generate synthetic source images
    3. Use a different dataset
    
    For this demo, we'll create a mapping file that can be used with
    a proper paired dataset.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load PIE-Bench mapping
    mapping_file = source_path / "mapping_file.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    with open(mapping_file) as f:
        pie_bench_data = json.load(f)
    
    print(f"Loaded {len(pie_bench_data)} entries from PIE-Bench")
    
    # Create training mapping
    # Note: PIE-Bench is designed for evaluation, not training
    # For actual training, you need a dataset with source/edited pairs
    
    training_mapping = []
    
    print("\n⚠️  WARNING: PIE-Bench only contains edited images!")
    print("For proper training, you need a dataset with source/edited pairs.")
    print("\nOptions:")
    print("1. Use InstructPix2Pix dataset (recommended)")
    print("2. Use MagicBrush dataset")
    print("3. Create synthetic pairs from COCO")
    print("\nFor now, creating a template mapping file...")
    
    # Create template mapping
    for idx, (key, entry) in enumerate(pie_bench_data.items()):
        # In a real training dataset, you would have:
        # - source_image: original image
        # - edited_image: edited version
        # - editing_instruction: what changed
        
        training_entry = {
            "id": key,
            "source_image": f"source/{entry['image_path']}",  # Would need actual source
            "edited_image": f"edited/{entry['image_path']}",
            "editing_instruction": entry.get("editing_instruction", ""),
            "original_prompt": entry.get("original_prompt", ""),
            "editing_prompt": entry.get("editing_prompt", ""),
        }
        training_mapping.append(training_entry)
    
    # Save mapping
    output_mapping = output_path / "mapping_file.json"
    with open(output_mapping, "w") as f:
        json.dump(training_mapping, f, indent=2)
    
    print(f"\n✅ Created template mapping: {output_mapping}")
    print(f"   Entries: {len(training_mapping)}")
    
    # Create README
    readme = output_path / "README.txt"
    with open(readme, "w") as f:
        f.write("PIE-Bench Training Preparation\n")
        f.write("=" * 50 + "\n\n")
        f.write("⚠️  IMPORTANT: PIE-Bench is designed for EVALUATION, not training.\n\n")
        f.write("For training, you need a dataset with source/edited image pairs.\n\n")
        f.write("Recommended datasets:\n")
        f.write("1. InstructPix2Pix: https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered\n")
        f.write("2. MagicBrush: https://huggingface.co/datasets/osunlp/MagicBrush\n")
        f.write("3. COCO + Synthetic edits\n\n")
        f.write("This script created a template mapping file.\n")
        f.write("You need to populate it with actual source/edited image pairs.\n")
    
    print(f"\n📄 Created README: {readme}")
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("\n1. Download a proper training dataset (InstructPix2Pix recommended)")
    print("2. Organize as source/edited pairs")
    print("3. Update mapping_file.json with actual paths")
    print("4. Run training script")
    print("\nFor quick demo, use MAE initialization without training!")


def download_instructpix2pix_sample():
    """
    Download a sample of InstructPix2Pix dataset for training.
    """
    print("\n" + "=" * 60)
    print("InstructPix2Pix Dataset Download")
    print("=" * 60)
    print("\nTo download InstructPix2Pix dataset:")
    print("\n1. Install datasets library:")
    print("   pip install datasets")
    print("\n2. Download dataset:")
    print("""
from datasets import load_dataset

# Load dataset (this will download ~25GB)
dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered")

# Save to disk
dataset.save_to_disk("./instructpix2pix_data")
""")
    print("\n3. The dataset contains:")
    print("   - input_image: source image")
    print("   - edited_image: edited image")
    print("   - edit_prompt: editing instruction")
    print("\n4. Convert to our format:")
    print("   python convert_instructpix2pix.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to PIE-Bench directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for prepared dataset")
    parser.add_argument("--show_download_info", action="store_true",
                        help="Show information about downloading proper training datasets")
    
    args = parser.parse_args()
    
    prepare_pie_bench(args.source_path, args.output_path)
    
    if args.show_download_info:
        download_instructpix2pix_sample()
    
    print("\n" + "=" * 60)
    print("⚡ QUICK START FOR TOMORROW:")
    print("=" * 60)
    print("\nSince you need results by tomorrow, SKIP TRAINING and use:")
    print("\n1. Initialize from MAE (10 minutes):")
    print("   python init_from_pretrained_vit.py \\")
    print("     --source-checkpoint mae_pretrain_vit_base.pth \\")
    print("     --out ./checkpoints/uvit_from_mae.pt \\")
    print("     --uvit_size mid")
    print("\n2. Run inference immediately (2 hours):")
    print("   python run_uvit_inference.py \\")
    print("     --checkpoint ./checkpoints/uvit_from_mae.pt \\")
    print("     --uvit_size mid \\")
    print("     --source_path ../benchmark/pie_bench \\")
    print("     --target_path ./outputs")
    print("\nThis will give you BETTER results than U-Net initialization!")
    print("=" * 60)
