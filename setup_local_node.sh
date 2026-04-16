#!/bin/bash

# 1. Load the Anaconda module
module load anaconda3

# 2. Initialize Conda for the script and activate your specific environment
# We use 'source' because 'conda activate' doesn't work in standard scripts without it
source /opt/packages/anaconda/anaconda3/etc/profile.d/conda.sh
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