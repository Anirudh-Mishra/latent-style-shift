#!/bin/bash
#
# Two-stage training script.
#
# Stage 1 (--distill): Knowledge distillation from the frozen LCM UNet.
#   The UNet already has strong text conditioning; the UViT learns to match
#   its noise predictions. This is far more sample-efficient than training
#   against raw noise from a cold MAE init.
#
# Stage 2 (fine-tune): Source-attention injection training on InstructPix2Pix.
#   The ptp_utils threshold bug is now fixed (32**2 -> 33**2), so the
#   StoredAttnInjector will actually receive stored maps this time.
#   Use a lower lr to preserve the text conditioning learned in Stage 1.
#

echo "Starting training script: resume_training.sh"
source /home/avid/latent-style-shift/.venv/bin/activate
echo "Activated virtual environment."

DATA_DIR=/home/avid/dl_data/instructpix2pix_50k/
MAE_CHECKPOINT=./checkpoints/uvit_from_mae.pt

STAGE1_OUT=./checkpoints/uvit_distill/
STAGE2_OUT=./checkpoints/uvit_finetuned/

# --- Stage 1: Knowledge distillation ---
echo "=========================================="
echo "Stage 1: Knowledge Distillation from LCM UNet"
echo "=========================================="

# Find latest stage-1 checkpoint for resume
if ls "$STAGE1_OUT"uvit_mid_epoch*.pt 1>/dev/null 2>&1; then
    STAGE1_RESUME=$(ls -t "$STAGE1_OUT"uvit_mid_epoch*.pt | head -1)
    echo "Resuming Stage 1 from: $STAGE1_RESUME"
else
    STAGE1_RESUME="$MAE_CHECKPOINT"
    echo "Starting Stage 1 fresh from MAE checkpoint"
fi

python train_uvit.py \
  --data_dir "$DATA_DIR" \
  --resume "$STAGE1_RESUME" \
  --distill \
  --uvit_size mid \
  --image_size 512 \
  --latent_size 64 \
  --patch_size 2 \
  --batch_size 4 \
  --num_epochs 5 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --max_grad_norm 1.0 \
  --num_workers 4 \
  --output_dir "$STAGE1_OUT" \
  --log_every 50 \
  --save_every 1 \
  --use_amp \
  --grad_accum_steps 4 \
  --seed 42

STAGE1_BEST="$STAGE1_OUT/uvit_mid_best.pt"
if [ ! -f "$STAGE1_BEST" ]; then
    echo "Stage 1 checkpoint not found at $STAGE1_BEST, aborting."
    exit 1
fi

# --- Stage 2: Source-attention injection fine-tune ---
echo ""
echo "=========================================="
echo "Stage 2: Source-Attention Fine-tuning"
echo "=========================================="

if ls "$STAGE2_OUT"uvit_mid_epoch*.pt 1>/dev/null 2>&1; then
    STAGE2_RESUME=$(ls -t "$STAGE2_OUT"uvit_mid_epoch*.pt | head -1)
    echo "Resuming Stage 2 from: $STAGE2_RESUME"
else
    STAGE2_RESUME="$STAGE1_BEST"
    echo "Starting Stage 2 from Stage 1 best: $STAGE2_RESUME"
fi

python train_uvit.py \
  --data_dir "$DATA_DIR" \
  --resume "$STAGE2_RESUME" \
  --uvit_size mid \
  --image_size 512 \
  --latent_size 64 \
  --patch_size 2 \
  --batch_size 4 \
  --num_epochs 5 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --warmup_steps 200 \
  --max_grad_norm 1.0 \
  --num_workers 4 \
  --output_dir "$STAGE2_OUT" \
  --log_every 50 \
  --save_every 1 \
  --use_amp \
  --grad_accum_steps 4 \
  --seed 42

echo ""
echo "=========================================="
echo "Training complete."
echo "  Stage 1 best: $STAGE1_OUT/uvit_mid_best.pt"
echo "  Stage 2 best: $STAGE2_OUT/uvit_mid_best.pt"
echo "Use Stage 2 best for inference."
echo "=========================================="
