#!/bin/bash
#
# Two-stage training script (optimized for RTX 5090 / Blackwell).
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
# Speed optimizations vs the naive script:
#   - Pre-encoded VAE latents + CLIP embeddings: saves ~70 ms/step
#   - BF16 autocast (no GradScaler overhead): native on Blackwell
#   - torch.compile max-autotune: ~20-40% kernel speedup
#   - batch_size=16, grad_accum=1: same effective batch, fewer small kernels
#

echo "Starting training script: resume_training.sh"
source /home/avid/latent-style-shift/.venv/bin/activate
echo "Activated virtual environment."

DATA_DIR=/home/avid/dl_data/instructpix2pix_50k/
ENCODED_DIR=/home/avid/dl_data/instructpix2pix_50k_encoded/
MAE_CHECKPOINT=./checkpoints/uvit_from_mae.pt

STAGE1_OUT=./checkpoints/uvit_distill/
STAGE2_OUT=./checkpoints/uvit_finetuned/

# --- 0. Pre-encode dataset (skips automatically if already done) ---
echo "=========================================="
echo "Step 0: Pre-encoding dataset to VAE latents + CLIP embeddings"
echo "=========================================="
python prepare_encoded_dataset.py \
    --data_dir "$DATA_DIR" \
    --out_dir  "$ENCODED_DIR" \
    --batch_size 256 \
    --num_workers 16

# # --- Stage 1: Knowledge distillation ---
# echo ""
# echo "=========================================="
# echo "Stage 1: Knowledge Distillation from LCM UNet"
# echo "=========================================="

# if ls "$STAGE1_OUT"uvit_mid_epoch*.pt 1>/dev/null 2>&1; then
#     STAGE1_RESUME=$(ls -t "$STAGE1_OUT"uvit_mid_epoch*.pt | head -1)
#     echo "Resuming Stage 1 from: $STAGE1_RESUME"
# else
#     STAGE1_RESUME="$MAE_CHECKPOINT"
#     echo "Starting Stage 1 fresh from MAE checkpoint"
# fi

# python train_uvit.py \
#   --encoded_data_dir "$ENCODED_DIR" \
#   --resume "$STAGE1_RESUME" \
#   --distill \
#   --uvit_size mid \
#   --latent_size 64 \
#   --patch_size 2 \
#   --batch_size 16 \
#   --num_epochs 10 \
#   --lr 1e-4 \
#   --weight_decay 0.01 \
#   --warmup_steps 500 \
#   --max_grad_norm 1.0 \
#   --num_workers 4 \
#   --output_dir "$STAGE1_OUT" \
#   --log_every 50 \
#   --save_every 1 \
#   --use_amp \
#   --bf16 \
#   --grad_accum_steps 1 \
#   --seed 42

# STAGE1_BEST="$STAGE1_OUT/uvit_mid_best.pt"
# if [ ! -f "$STAGE1_BEST" ]; then
#     echo "Stage 1 checkpoint not found at $STAGE1_BEST, aborting."
#     exit 1
# fi

# # --- Stage 2: Source-attention injection fine-tune ---
# echo ""
# echo "=========================================="
# echo "Stage 2: Source-Attention Fine-tuning"
# echo "=========================================="

# if ls "$STAGE2_OUT"uvit_mid_epoch*.pt 1>/dev/null 2>&1; then
#     STAGE2_RESUME=$(ls -t "$STAGE2_OUT"uvit_mid_epoch*.pt | head -1)
#     STAGE2_RESET=""
#     echo "Resuming Stage 2 from: $STAGE2_RESUME"
# else
#     STAGE2_RESUME="$STAGE1_BEST"
#     STAGE2_RESET="--reset_epoch"
#     echo "Starting Stage 2 fresh from Stage 1 best: $STAGE2_RESUME"
# fi

# python train_uvit.py \
#   $STAGE2_RESET \
#   --distill \
#   --encoded_data_dir "$ENCODED_DIR" \
#   --resume "$STAGE2_RESUME" \
#   --uvit_size mid \
#   --latent_size 64 \
#   --patch_size 2 \
#   --batch_size 16 \
#   --num_epochs 15 \
#   --lr 2e-5 \
#   --weight_decay 0.01 \
#   --warmup_steps 6250 \
#   --max_grad_norm 1.0 \
#   --num_workers 4 \
#   --output_dir "$STAGE2_OUT" \
#   --log_every 50 \
#   --save_every 1 \
#   --use_amp \
#   --bf16 \
#   --grad_accum_steps 1 \
#   --seed 42 \
#   --inject_alpha 0.4 \
#   --edit_loss_weight 0.5

STAGE2_BEST="$STAGE2_OUT/uvit_mid_best.pt"

# =====================================================================
# OPTION B: UNet cross-attention init + three training patches
# =====================================================================
# Step 1: Build hybrid checkpoint (MAE self-attn + UNet cross-attn)
# Step 2: Stage A — distillation only, let cross-attn settle into network
# Step 3: Stage B — full edit-direction training with all three patches
# =====================================================================

OPTB_INIT=./checkpoints/uvit_mae_self_unet_cross.pt
OPTB_STAGE_A_OUT=./checkpoints/uvit_optb_stage_a/
OPTB_STAGE_B_OUT=./checkpoints/uvit_optb_stage_b/

# --- Step 1: Build hybrid init (~2 minutes) ---
echo ""
echo "=========================================="
echo "Option B Step 1: Building hybrid init checkpoint"
echo "=========================================="

if [ ! -f "$OPTB_INIT" ]; then
    python init_from_unet_cross_only.py \
        --mae_checkpoint ./checkpoints/uvit_from_mae.pt \
        --out "$OPTB_INIT" \
        --uvit_size mid \
        --img_size 64 \
        --patch_size 2
else
    echo "Hybrid init already exists at $OPTB_INIT — skipping."
fi

if [ ! -f "$OPTB_INIT" ]; then
    echo "ERROR: hybrid init was not produced. Aborting."
    exit 1
fi

# --- Step 2: Stage A — distillation only (~30 minutes) ---
# Lets cross-attention adapt to the dimensional mismatch from bilinear resize.
# CFG dropout is on so the model learns text/null separation from step 0,
# but no edit_loss or same-latent forcing yet — keep it gentle while the
# resized cross-attn weights find their footing alongside MAE self-attn.
echo ""
echo "=========================================="
echo "Option B Step 2: Stage A — distillation"
echo "=========================================="

python train_uvit.py \
  --reset_epoch \
  --distill \
  --encoded_data_dir "$ENCODED_DIR" \
  --resume "$OPTB_INIT" \
  --uvit_size mid \
  --latent_size 64 \
  --patch_size 2 \
  --batch_size 16 \
  --num_epochs 8 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --max_grad_norm 1.0 \
  --num_workers 4 \
  --output_dir "$OPTB_STAGE_A_OUT" \
  --log_every 50 \
  --save_every 1 \
  --use_amp \
  --bf16 \
  --grad_accum_steps 1 \
  --seed 42 \
  --inject_alpha 0.0 \
  --edit_loss_weight 0.0 \
  --same_latent_prob 0.0 \
  --cfg_dropout_prob 0.10

OPTB_STAGE_A_BEST="$OPTB_STAGE_A_OUT/uvit_mid_best.pt"
if [ ! -f "$OPTB_STAGE_A_BEST" ]; then
    echo "ERROR: Stage A did not produce a best checkpoint. Aborting."
    exit 1
fi

# --- Step 3: Stage B — full edit-direction training (~45 minutes) ---
# All three bug fixes active. Lower lr to refine without destroying Stage A.
echo ""
echo "=========================================="
echo "Option B Step 3: Stage B — edit-direction training"
echo "=========================================="

python train_uvit.py \
  --reset_epoch \
  --distill \
  --encoded_data_dir "$ENCODED_DIR" \
  --resume "$OPTB_STAGE_A_BEST" \
  --uvit_size mid \
  --latent_size 64 \
  --patch_size 2 \
  --batch_size 16 \
  --num_epochs 8 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --max_grad_norm 1.0 \
  --num_workers 4 \
  --output_dir "$OPTB_STAGE_B_OUT" \
  --log_every 50 \
  --save_every 1 \
  --use_amp \
  --bf16 \
  --grad_accum_steps 1 \
  --seed 42 \
  --inject_alpha 0.0 \
  --edit_loss_weight 0.7 \
  --same_latent_prob 1.0 \
  --cfg_dropout_prob 0.10

echo ""
echo "=========================================="
echo "Option B training complete."
echo "  Stage A best: $OPTB_STAGE_A_OUT/uvit_mid_best.pt"
echo "  Stage B best: $OPTB_STAGE_B_OUT/uvit_mid_best.pt"
echo "Use Stage B best for inference."
echo "=========================================="
