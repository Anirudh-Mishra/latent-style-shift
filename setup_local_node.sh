#!/bin/bash

# 1. Load the Anaconda module
module load anaconda3

# 2. Initialize Conda for the script and activate your specific environment
conda deactivate  # Ensures we exit the base environment first
conda activate /ocean/projects/cis260127p/shared/UNetENV

# 3. Handle the Data (Unzip to /$LOCAl)
echo "Getting data to /local/data..."
mkdir -p $LOCAL/data/ && cd $LOCAL/data/
mkdir -p $LOCAL/data/train2017 $LOCAL/data/val2017 $LOCAL/data/annotations
unzip -oq /ocean/projects/cis260127p/shared/data/Project/train2017.zip -d $LOCAL/data/train2017
unzip -oq /ocean/projects/cis260127p/shared/data/Project/val2017.zip   -d $LOCAL/data/val2017
cp /ocean/projects/cis260127p/shared/data/Project/annotations/captions_train2017.json $LOCAL/data/annotations/
cp /ocean/projects/cis260127p/shared/data/Project/annotations/captions_val2017.json   $LOCAL/data/annotations/

# 4. Load the GPU Drivers/Libraries
module load cudnn
module load cuda

echo "Setup complete! Environment and data are ready."
USER_DIR=$(whoami)
COCO_MAIN_DIR="${LOCAL}/data/"
COCO_ANNOTATIONS="${LOCAL}/data/annotations/captions_train2017.json"
COCO_IMAGES_DIR="${LOCAL}/data/train2017"
COCO_OUT_DIR="/ocean/projects/cis260127p/${USER_DIR}/outputs/uvit_ablation_batchsize"
COCO_BEST_DIR="/ocean/projects/cis260127p/${USER_DIR}/outputs/uvit_mid_seed42_edit_instruction/best_model.pt"

echo "Running training inference script now..."
python /jet/home/adube1/latent-style-shift/model/train_uvit.py \
    --data_dir "$COCO_MAIN_DIR" \
    --coco_annotations "$COCO_ANNOTATIONS" \
    --coco_images_dir "$COCO_IMAGES_DIR" \
    --uvit_size mid \
    --patch_size 2 \
    --image_size 512 \
    --latent_size 64 \
    --batch_size 4 \
    --num_epochs 4 \
    --lr 1e-4 \
    --grad_accum_steps 8 \
    --use_amp \
    --output_dir "$COCO_OUT_DIR"
    # --resume_from_checkpoint "$COCO_BEST_DIR" # add only if wanting to resume from a previous checkpoint