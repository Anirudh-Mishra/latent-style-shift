import os
import argparse
import glob
import subprocess
import sys

"""
Evaluate a set of UViT checkpoints: for each checkpoint, run inference (producing outputs)
and run the evaluator to produce CSVs. Collected CSVs are written into `results/` under the
experiment directory.

Usage:
  python model/eval_checkpoints.py --ckpt_dir out/uvit_checkpoints --data_dir data --results_dir results

Requires `model/run_uvit_inference.py` and `eval/evaluation/evaluate.py` to be present.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--python', type=str, default=sys.executable)
    parser.add_argument('--uvit_size', type=str, default='mid')
    parser.add_argument('--infer_script', type=str, default=os.path.join('model','run_uvit_inference.py'))
    parser.add_argument('--eval_script', type=str, default=os.path.join('eval','evaluation','evaluate.py'))
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, '*.pt')))
    if len(ckpts) == 0:
        print('No checkpoints found in', args.ckpt_dir)
        return

    for ckpt in ckpts:
        name = os.path.splitext(os.path.basename(ckpt))[0]
        out_dir = os.path.join('output', name)
        print('Running inference for', ckpt, '->', out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # run inference wrapper
        cmd_inf = [args.python, args.infer_script, '--checkpoint', ckpt, '--uvit_size', args.uvit_size,
                   '--source_path', args.data_dir, '--target_path', out_dir]
        r = subprocess.run(cmd_inf)
        if r.returncode != 0:
            print('Inference failed for', ckpt)
            continue

        # run evaluator
        csv_out = os.path.join(args.results_dir, f'{name}.csv')
        cmd_eval = [args.python, args.eval_script, '--annotation_mapping_file', os.path.join(args.data_dir,'mapping_file.json'),
                    '--src_image_folder', os.path.join(args.data_dir,'annotation_images'), '--result_path', csv_out]
        r2 = subprocess.run(cmd_eval)
        if r2.returncode != 0:
            print('Evaluation failed for', ckpt)
            continue
        print('Wrote metrics to', csv_out)


if __name__ == '__main__':
    main()
