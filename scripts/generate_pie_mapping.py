#!/usr/bin/env python3
"""
Generate a PIE-Bench `mapping_file.json` for evaluation scripts.

Writes entries keyed by relative image path (relative to `annotation_images`) with
fields: `image_path`, `editing_prompt`, and `mask` (None by default).

Usage:
  python scripts/generate_pie_mapping.py --pie_root /path/to/PIE-Bench_v1

Optional: populate prompts from sidecar `.txt` files if present.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pie_root", type=str, required=True, help="Path to PIE-Bench root")
    p.add_argument("--annotation_dir", type=str, default="annotation_images",
                   help="Directory under pie_root that contains annotated images")
    p.add_argument("--out", type=str, default=None, help="Output mapping_file.json path")
    p.add_argument("--use_txt_prompts", action="store_true",
                   help="If set, read per-image prompt from sidecar .txt files (same basename)")
    args = p.parse_args()

    root = Path(args.pie_root)
    ann_dir = root / args.annotation_dir
    if not ann_dir.exists():
        # fall back to root as the images directory
        ann_dir = root

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    mapping = {}

    for img_path in sorted(ann_dir.rglob("*")):
        if img_path.suffix.lower() in exts and img_path.is_file():
            try:
                rel = img_path.relative_to(ann_dir).as_posix()
            except Exception:
                rel = img_path.name

            prompt = ""
            if args.use_txt_prompts:
                txt = img_path.with_suffix(".txt")
                if not txt.exists():
                    # also check sidecar in same folder but with base name
                    alt = img_path.with_suffix("")
                    txt = alt.with_suffix(".txt")
                if txt.exists():
                    try:
                        prompt = txt.read_text(encoding="utf-8").strip()
                    except Exception:
                        prompt = ""

            mapping[rel] = {"image_path": rel, "editing_prompt": prompt, "mask": None}

    out_path = Path(args.out) if args.out is not None else root / "mapping_file.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with {len(mapping)} entries")


if __name__ == "__main__":
    main()
