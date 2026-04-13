import os
import argparse
import subprocess
import sys

"""
Simple wrapper to run the existing model/run_pie_bench.py using U-ViT adapter.
It sets environment variables expected by `run_pie_bench.py`:
- BACKBONE=uvit
- UVIT_CHECKPOINT=<path>
- UVIT_SIZE=<preset>

Then invokes the script to generate outputs in the given target directory.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to UViT checkpoint (.pt)")
    parser.add_argument("--uvit_size", type=str, default="mid", choices=["small","mid","large"])
    parser.add_argument("--source_path", type=str, required=True, help="PIE-Bench root (contains mapping_file.json and annotation_images)")
    parser.add_argument("--target_path", type=str, required=True, help="Output root where annotation_images will be written")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to run the bench script")
    # forward inference flags to run_pie_bench.py
    parser.add_argument('--num_inference_steps', type=int, default=12)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--cross_replace_steps', type=float, default=0.7)
    parser.add_argument('--self_replace_steps', type=float, default=0.7)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--thresh_e', type=float, default=0.55)
    parser.add_argument('--thresh_m', type=float, default=0.6)
    parser.add_argument('--denoise', action='store_true')

    args = parser.parse_args()

    env = os.environ.copy()
    env["BACKBONE"] = "uvit"
    env["UVIT_CHECKPOINT"] = os.path.abspath(args.checkpoint)
    env["UVIT_SIZE"] = args.uvit_size

    bench_script = os.path.join(os.path.dirname(__file__), "run_pie_bench.py")

    cmd = [args.python, bench_script, "--source_path", args.source_path, "--target_path", args.target_path,
           "--num_inference_steps", str(args.num_inference_steps),
           "--strength", str(args.strength),
           "--cross_replace_steps", str(args.cross_replace_steps),
           "--self_replace_steps", str(args.self_replace_steps),
           "--eta", str(args.eta),
           "--thresh_e", str(args.thresh_e),
           "--thresh_m", str(args.thresh_m)]
    if args.denoise:
        cmd.append("--denoise")

    print("Running inference with UViT via:", " ".join(cmd))
    print("Using checkpoint:", env["UVIT_CHECKPOINT"]) 

    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print("run_pie_bench.py failed with exit code", proc.returncode)
        sys.exit(proc.returncode)
    else:
        print("Inference completed. Outputs written to:", args.target_path)


if __name__ == "__main__":
    main()
