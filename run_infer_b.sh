#!/bin/bash

source /home/avid/latent-style-shift/.venv/bin/activate

if [ ! -d "/home/avid/dl_data/PIE-Bench_v1" ]; then
    echo "Data not found. Unzipping now..."
    unzip -oq PIE-Bench_v1-20260416T022042Z-3-001.zip -d /home/avid/dl_data
else
    echo "Data already exists. Skipping unzip."
fi

echo "Starting inferencing (Option B checkpoint)..."

python model/run_uvit_inference.py \
    --checkpoint  /home/avid/latent-style-shift/model/checkpoints/uvit_optb_stage_b/uvit_mid_best.pt \
    --source_path /home/avid/dl_data/PIE-Bench_v1/ \
    --target_path /home/avid/dl_data/outputs/uvit_optb_outputs/ \
    --denoise \
    --guidance_t 1.5 \
    --guidance_s 1.0 \
    --cross_replace_steps 0.4 \
    --self_replace_steps 0.4 \
    --patch_smooth_sigma 0.7

echo "Process complete."
