#!/usr/bin/env python3
"""Prepare COCO captions JSON for `train_uvit.py`.

Writes `mapping_file.json` into `--out_dir` (default: data_dir) with absolute image paths.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from train_uvit import prepare_coco_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_annotations", type=str, required=True)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    images_dir = args.images_dir
    if images_dir is None:
        annp = Path(args.coco_annotations)
        cand = annp.parent.parent / "train2017"
        if cand.exists():
            images_dir = str(cand)
        else:
            raise ValueError("--images_dir must be provided or train2017 must be next to annotations")

    prepare_coco_mapping(args.coco_annotations, images_dir, args.out_dir)


if __name__ == "__main__":
    main()
