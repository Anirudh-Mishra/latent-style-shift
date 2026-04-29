#!/bin/bash
#
# Aggressive training with CFG dropout
# Auto-detects latest checkpoint and resumes
# Higher LR + higher CFG dropout to force text conditioning fast
#

OUTPUT_DIR="./checkpoints/uvit_trained_50k"

LATEST_EPOCH=0
RESUME_FROM=""
for f in "$OUTPUT_DIR"/uvit_mid_epoch*.pt; do
    [ -f "$f" ] || continue
    epoch_num=$(echo "$f" | grep -oP 'epoch\K[0-9]+')
    if [ "$epoch_num" -gt "$LATEST_EPOCH" ]; then
        LATEST_EPOCH=$epoch_num
        RESUME_FROM="$f"
    fi
done

if [ -z "$RESUME_FROM" ]; then
    echo "ERROR: No epoch checkpoints found in $OUTPUT_DIR"
    exit 1
fi

echo "=========================================="
echo "AGGRESSIVE U-ViT Training with CFG Dropout"
echo "Resume from: $RESUME_FROM (epoch $LATEST_EPOCH)"
echo "CFG dropout: 15% (aggressive)"
echo "Learning rate: 2e-4 (4x higher - spatial already converged)"
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
  --num_epochs $((LATEST_EPOCH + 10)) \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --warmup_steps 200 \
  --max_grad_norm 1.0 \
  --cfg_dropout_prob 0.15 \
  --num_workers 4 \
  --output_dir "$OUTPUT_DIR" \
  --log_every 100 \
  --save_every 1 \
  --use_amp \
  --grad_accum_steps 4 \
  --seed 42

echo ""
echo "=========================================="
echo "Training complete!"
echo "Test with: python run_pie_bench.py --backbone uvit --uvit_checkpoint $OUTPUT_DIR/uvit_mid_best.pt --source_path ../benchmark/PIE-Bench_v1 --target_path ./outputs_uvit_cfg --guidance_t 3.0 --num_inference_steps 15"
echo "=========================================="
