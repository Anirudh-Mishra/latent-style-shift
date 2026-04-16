#!/bin/bash

# 1. Load the Anaconda module
module load anaconda3

# 2. Initialize Conda for the script and activate your specific environment
conda deactivate  # Ensures we exit the base environment first
conda activate /ocean/projects/cis260127p/shared/UNetENV

# 3. Handle the Data (Unzip to /tmp)
# We use -o to overwrite and -q for quiet so the terminal doesn't lag
echo "Unzipping data to /tmp/UNet_Data..."
unzip -oq PIE-Bench_v1-20260416T022042Z-3-001.zip -d /tmp/UNet_Data

# 4. Load the GPU Drivers/Libraries
module load cudnn
module load cuda

echo "Setup complete! Environment and data are ready."

python model/train_uvit.py   --data_dir /tmp/UNet_Data/PIE-Bench_v1/   --uvit_size mid   --batch_size 8   --num_epochs 100   --lr 1e-4   --output_dir /ocean/projects/cis260127p/adube1/outputs/uvit_mid_seed42_edit_instruction/   --seed 42   --use_amp

python model/run_uvit_inference.py --checkpoint /ocean/projects/cis260127p/adube1/outputs/uvit_mid_seed42_edit_instruction/uvit_mid_best.pt --source_path /tmp/UNet_Data/PIE-Bench_v1/ --target_path /ocean/projects/cis260127p/adube1/outputs/uvit_mid_seed42_eval_edit_instruction
