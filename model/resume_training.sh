#!/bin/bash
#
# Smart resume script - automatically finds latest checkpoint
# Safe to run multiple times if PSC disconnects (8hr limit)
#
# USAGE:
#   First run:   starts from MAE init checkpoint
#   Later runs:  auto-detects the latest epoch checkpoint and resumes
#

OUTPUT_DIR="./checkpoints/uvit_trained_50k"
INITIAL_CHECKPOINT="./checkpoints/uvit_from_mae.pt"

# Find the latest checkpoint
if [ -f "$OUTPUT_DIR/uvit_mid_epoch5.pt" ]; then
    echo "✅ Training already complete (epoch 5 found)"
    echo "Best checkpoint: $OUTPUT_DIR/uvit_mid_best.pt"
    exit 0
elif [ -f "$OUTPUT_DIR/uvit_mid_epoch4.pt" ]; then
    RESUME_FROM="$OUTPUT_DIR/uvit_mid_epoch4.pt"
    echo "📂 Resuming from epoch 4"
elif [ -f "$OUTPUT_DIR/uvit_mid_epoch3.pt" ]; then
    RESUME_FROM="$OUTPUT_DIR/uvit_mid_epoch3.pt"
    echo "📂 Resuming from epoch 3"
elif [ -f "$OUTPUT_DIR/uvit_mid_epoch2.pt" ]; then
    RESUME_FROM="$OUTPUT_DIR/uvit_mid_epoch2.pt"
    echo "📂 Resuming from epoch 2"
elif [ -f "$OUTPUT_DIR/uvit_mid_epoch1.pt" ]; then
    RESUME_FROM="$OUTPUT_DIR/uvit_mid_epoch1.pt"
    echo "📂 Resuming from epoch 1"
else
    RESUME_FROM="$INITIAL_CHECKPOINT"
    echo "🆕 Starting fresh from MAE checkpoint"
fi

echo "=========================================="
echo "U-ViT Training on InstructPix2Pix 50K"
echo "Resume from: $RESUME_FROM"
echo "=========================================="
echo ""

python train_uvit.py \
  --data_dir ./data/instructpix2pix_50k \
  --resume "$RESUME_FROM" \
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
  --output_dir "$OUTPUT_DIR" \
  --log_every 100 \
  --save_every 1 \
  --use_amp \
  --grad_accum_steps 4 \
  --seed 42

echo ""
echo "=========================================="
echo "Training session complete!"
echo "Check: $OUTPUT_DIR"
echo "=========================================="
