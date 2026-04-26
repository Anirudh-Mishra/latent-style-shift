import os
import argparse
import subprocess
import sys

"""
Simple wrapper to run the existing model/run_pie_bench.py using U-ViT adapter.
It forwards U-ViT settings through CLI arguments instead of environment variables.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to UViT checkpoint (.pt)")
    parser.add_argument("--uvit_size", type=str, default="mid", choices=["small", "mid", "large"])
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size used when training the checkpoint")
    parser.add_argument("--source_path", type=str, required=True, help="PIE-Bench root containing mapping_file.json and annotation_images")
    parser.add_argument("--target_path", type=str, required=True, help="Output root where annotation_images will be written")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to run the bench script")

    parser.add_argument("--num_inference_steps", type=int, default=12)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--cross_replace_steps", type=float, default=0.7)
    parser.add_argument("--self_replace_steps", type=float, default=0.7)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--thresh_e", type=float, default=0.55)
    parser.add_argument("--thresh_m", type=float, default=0.6)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--guidance_t", type=float, default=7.5,
                        help="Target guidance scale (UViT default 7.5 vs UNet default 2.3)")
    parser.add_argument("--guidance_s", type=float, default=1.0,
                        help="Source guidance scale")
    parser.add_argument("--source_conditioned", action="store_true",
                        help="Enable source-conditioned UViT; must match training/init")

    args = parser.parse_args()

    bench_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_pie_bench.py")
    if not os.path.exists(bench_script):
        print(f"ERROR: run_pie_bench.py not found at {bench_script}")
        print("Make sure run_uvit_inference.py and run_pie_bench.py are in the same directory.")
        sys.exit(1)

    cmd = [
        args.python, bench_script,
        "--backbone", "uvit",
        "--uvit_checkpoint", os.path.abspath(args.checkpoint),
        "--uvit_size", args.uvit_size,
        "--uvit_patch_size", str(args.patch_size),
        "--source_path", args.source_path,
        "--target_path", args.target_path,
        "--num_inference_steps", str(args.num_inference_steps),
        "--strength", str(args.strength),
        "--cross_replace_steps", str(args.cross_replace_steps),
        "--self_replace_steps", str(args.self_replace_steps),
        "--eta", str(args.eta),
        "--thresh_e", str(args.thresh_e),
        "--thresh_m", str(args.thresh_m),
        "--guidance_t", str(args.guidance_t),
        "--guidance_s", str(args.guidance_s),
    ]
    if args.denoise:
        cmd.append("--denoise")
    if args.source_conditioned:
        cmd.append("--source_conditioned")

    print("Running inference with UViT via:", " ".join(cmd))
    print("Using checkpoint:", os.path.abspath(args.checkpoint))

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print("run_pie_bench.py failed with exit code", proc.returncode)
        sys.exit(proc.returncode)

    print("Inference completed. Outputs written to:", args.target_path)


if __name__ == "__main__":
    main()
