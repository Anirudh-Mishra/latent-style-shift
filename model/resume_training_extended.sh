#!/bin/bash
#
# Extended training - resume from epoch 5 checkpoint
# Target: 50 total epochs (45 more) for visible editing results
#
# On g6e.xlarge (L40S): ~18 hours
# On g5.xlarge  (A10G): ~35 hours
#

OUTPUT_DIR="./checkpoints/uvit_trained_50k"
RESUME_FROM="$OUTPUT_DIR/uvit_mid_epoch8.pt"

if [ ! -f "$RESUME_FROM" ]; then
    echo "ERROR: Cannot find epoch 6 checkpoint at $RESUME_FROM"
    echo "Looking for available checkpoints..."
    ls -la "$OUTPUT_DIR"/*.pt 2>/dev/null
    exit 1
fi

echo "=========================================="
echo "U-ViT Extended Training (epochs 6-50)"
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
  --num_epochs 50 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_steps 0 \
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
echo "Extended training complete!"
echo "Use checkpoint: $OUTPUT_DIR/uvit_mid_best.pt"
echo "=========================================="
