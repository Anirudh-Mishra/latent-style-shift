#!/usr/bin/env python3
"""Thin wrapper that invokes the canonical benchmark evaluator so results are identical.

Usage:
  python3 tools/eval_pie_fast.py --generated_root outputs/uvit_pie

This will run: benchmark/evaluate_pie_bench.py --source_path /tmp/UNet_Data/PIE-Bench_v1 --output_path <generated_root>
"""
import argparse
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mapping', default='/tmp/UNet_Data/PIE-Bench_v1/mapping_file.json')
    p.add_argument('--source_root', default='/tmp/UNet_Data/PIE-Bench_v1')
    p.add_argument('--generated_root', required=True,
                   help='Path to generated outputs (contains annotation_images)')
    p.add_argument('--save_csv', default=None,
                   help='Optional custom CSV path (default: <generated_root>/piebench_table1_metrics.csv)')
    args, unknown = p.parse_known_args()

    save_csv = args.save_csv or os.path.join(args.generated_root, 'piebench_table1_metrics.csv')

    cmd = [sys.executable, 'benchmark/evaluate_pie_bench.py',
           '--source_path', args.source_root,
           '--output_path', args.generated_root,
           '--save_csv', save_csv]

    # Forward any extra flags to the canonical script
    if len(unknown) > 0:
        cmd += unknown

    print('Running canonical evaluator:')
    print(' '.join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f'benchmark/evaluate_pie_bench.py exited with {rc}')


if __name__ == '__main__':
    main()
