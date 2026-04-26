#!/bin/bash

# --- 1. SETUP ENVIRONMENT ---
# echo "Loading environment and drivers..."
# module load anaconda3
# module load cuda
# module load cudnn

# # Initialize Conda for the script
# conda deactivate
# conda activate /ocean/projects/cis260127p/shared/UNetENV
source /home/avid/latent-style-shift/.venv/bin/activate

# --- 2. SETUP LOCAL DATA ---
# Check if data is already unzipped in /tmp to save time
if [ ! -d "/home/avid/dl_data/PIE-Bench_v1" ]; then
    echo "Data not found in /tmp. Unzipping now..."
    # Replace the filename below if your zip name is different
    unzip -oq PIE-Bench_v1-20260416T022042Z-3-001.zip -d /home/avid/dl_data
else
    echo "Data already exists in /tmp. Skipping unzip."
fi

echo "Starting inferencing..."

python model/run_uvit_inference.py \
    --checkpoint  /home/avid/latent-style-shift/model/checkpoints/uvit_finetuned/uvit_mid_best.pt \
    --source_path /home/avid/dl_data/PIE-Bench_v1/ \
    --target_path /home/avid/dl_data/outputs/uvit_test_outputs/ \
    --denoise \
    --guidance_t 7.5 \
    --guidance_s 1.0

echo "Process complete."