#!/bin/bash

# --- 1. SETUP ENVIRONMENT ---
echo "Loading environment and drivers..."
module load anaconda3
module load cuda
module load cudnn

# Initialize Conda for the script
conda deactivate
conda activate /ocean/projects/cis260127p/shared/UNetENV

# --- 2. SETUP LOCAL DATA ---
# Check if data is already unzipped in /tmp to save time
if [ ! -d "/tmp/UNet_Data/PIE-Bench_v1" ]; then
    echo "Data not found in /tmp. Unzipping now..."
    # Replace the filename below if your zip name is different
    unzip -oq PIE-Bench_v1-20260416T022042Z-3-001.zip -d /tmp/UNet_Data
else
    echo "Data already exists in /tmp. Skipping unzip."
fi

# --- 3. RUN THE MODEL ---
echo "Starting training..."

# I used the cleaned version of your command
# Note: I'm saving outputs to Ocean so they aren't lost when the node resets
python model/train_uvit.py \
  --data_dir /tmp/UNet_Data/PIE-Bench_v1/ \
  --uvit_size mid \
  --batch_size 8 \
  --num_epochs 100 \
  --lr 1e-4 \
  --output_dir /ocean/projects/cis260127p/adube1/outputs/uvit_mid_seed42_edit_instruction \
  --seed 42 \
  --use_amp


echo "Starting inferencing..."

python model/run_uvit_inference.py \
    --checkpoint /ocean/projects/cis260127p/adube1/outputs/uvit_mid_seed42_edit_instruction/uvit_mid_best.pt \
    --source_path /tmp/UNet_Data/PIE-Bench_v1/ \
    --target_path /ocean/projects/cis260127p/adube1/outputs/uvit_mid_seed42_eval_edit_instruction

echo "Process complete."