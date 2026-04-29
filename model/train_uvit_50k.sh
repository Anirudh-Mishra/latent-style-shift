#!/bin/bash
#
# Training script for U-ViT on InstructPix2Pix 50K dataset
# Optimized for ~24h training window
#
# IMPORTANT: Before running, you must re-initialize the MAE checkpoint
# with the fixed code:
#
#   python init_from_pretrained_vit.py \
#     --source-checkpoint <path/to/mae_pretrain_vit_base.pth> \
#     --out ./checkpoints/uvit_from_mae.pt \
#     --img_size 64 --patch_size 2 --in_chans 4
#

echo "=========================================="
echo "U-ViT Training on InstructPix2Pix 50K"
echo "=========================================="
echo ""
echo "Dataset: 50,000 samples"
echo "Epochs: 5"
echo "Expected time: 8-12 hours"
echo ""

python train_uvit.py \
  --data_dir ./data/instructpix2pix_50k \
  --resume ./checkpoints/uvit_from_mae.pt \
  --uvit_size mid \
  --image_size 512 \
  --latent_size 64 \
  --patch_size 2 \
  --batch_size 4 \
  --num_epochs 5 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --max_grad_norm 1.0 \
  --num_workers 4 \
  --output_dir ./checkpoints/uvit_trained_50k \
  --log_every 100 \
  --save_every 1 \
  --use_amp \
  --grad_accum_steps 4 \
  --seed 42

echo ""
echo "=========================================="
echo "Training complete!"
echo "Best checkpoint: ./checkpoints/uvit_trained_50k/uvit_mid_best.pt"
echo "=========================================="
