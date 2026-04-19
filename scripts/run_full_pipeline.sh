#!/usr/bin/env bash
set -euo pipefail

# Usage: adjust defaults below or provide env vars.
# Example:
# COCO_ANN=data/annotations/captions_train2017.json COCO_IMGS=data/train2017 \ 
# TARGET_STEPS=100000 BATCH_SIZE=8 ./scripts/run_full_pipeline.sh

#######################
# Configurable params
#######################
CONDA_ENV="/ocean/projects/cis260127p/shared/UNetENV"
INSTALL_DEPS=false

# COCO inputs
: ${COCO_ANN:=data/annotations/captions_train2017.json}
: ${COCO_IMGS:=data/train2017}
: ${COCO_OUT:=/tmp/UNet_Data/COCO_for_training}

# training params (you can change TARGET_STEPS or NUM_EPOCHS)
: ${UVIT_SIZE:=mid}
: ${BATCH_SIZE:=8}
: ${TARGET_STEPS:=100000}
: ${NUM_EPOCHS:=""}
: ${LR:=1e-4}
: ${OUTPUT_DIR:=/ocean/projects/cis260127p/$USER/outputs/uvit_${UVIT_SIZE}}
: ${USE_AMP:=true}
: ${SAVE_EVERY:=5}

# PIE-Bench inputs/outputs
: ${PIE_ROOT:=/tmp/UNet_Data/PIE-Bench_v1}
: ${BASELINE_OUT:=./outputs/pie_baseline_unet}
: ${UVIT_OUT:=./outputs/pie_uvit}

mkdir -p "$OUTPUT_DIR"
mkdir -p "$BASELINE_OUT"
mkdir -p "$UVIT_OUT"

echo "Activating conda env: $CONDA_ENV"
conda deactivate || true
conda activate "$CONDA_ENV"

if [ "$INSTALL_DEPS" = true ]; then
  echo "Installing python deps (this may take a while)"
  pip install -r model/requirements.txt
  pip install -r benchmark/requirements.txt
  pip install open-clip-torch lpips scikit-image pandas tqdm
fi

echo "Preparing COCO mapping (out: $COCO_OUT)"
mkdir -p "$COCO_OUT"
python scripts/prepare_coco_for_uvit.py --coco_annotations "$COCO_ANN" --images_dir "$COCO_IMGS" --out_dir "$COCO_OUT"

# Count samples in mapping
NUM_SAMPLES=$(python - <<'PY'
import json,sys
m=json.load(open("$COCO_OUT/mapping_file.json"))
print(len(m))
PY
)
echo "Found $NUM_SAMPLES samples in mapping"

# compute steps/epoch and epochs if not explicitly set
STEPS_PER_EPOCH=$(( (NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))
if [ -z "$NUM_EPOCHS" ]; then
  NUM_EPOCHS=$(( (TARGET_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))
fi

echo "Training config: samples=$NUM_SAMPLES batch=$BATCH_SIZE steps_per_epoch=$STEPS_PER_EPOCH target_steps=$TARGET_STEPS epochs=$NUM_EPOCHS"

TRAIN_CMD=(python model/train_uvit.py --data_dir "$COCO_OUT" --uvit_size "$UVIT_SIZE" --batch_size "$BATCH_SIZE" --num_epochs "$NUM_EPOCHS" --lr "$LR" --output_dir "$OUTPUT_DIR" --seed 42 --save_every "$SAVE_EVERY")
if [ "$USE_AMP" = true ]; then
  TRAIN_CMD+=(--use_amp)
fi
echo "Running training: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

# locate best checkpoint
CKPT="$OUTPUT_DIR/uvit_${UVIT_SIZE}_best.pt"
if [ ! -f "$CKPT" ]; then
  echo "Best checkpoint not found at $CKPT; attempting to find latest epoch checkpoint"
  LATEST=$(ls -1t "$OUTPUT_DIR"/uvit_${UVIT_SIZE}_epoch*.pt 2>/dev/null | head -n1 || true)
  if [ -n "$LATEST" ]; then
    CKPT="$LATEST"
    echo "Using latest checkpoint: $CKPT"
  else
    echo "No checkpoint found in $OUTPUT_DIR" >&2
    exit 2
  fi
fi

echo "Running baseline UNet PIE-Bench inference (produces $BASELINE_OUT)"
python benchmark/run_pie_bench.py --source_path "$PIE_ROOT" --target_path "$BASELINE_OUT" --num_inference_steps 12 --strength 1.0 --cross_replace_steps 0.7 --self_replace_steps 0.7

echo "Running U-ViT PIE-Bench inference (produces $UVIT_OUT) with checkpoint $CKPT"
python model/run_uvit_inference.py --checkpoint "$CKPT" --uvit_size "$UVIT_SIZE" --source_path "$PIE_ROOT" --target_path "$UVIT_OUT" --num_inference_steps 12 --strength 1.0 --cross_replace_steps 0.7 --self_replace_steps 0.7

echo "Evaluating baseline outputs"
python benchmark/evaluate_pie_bench.py --source_path "$PIE_ROOT" --output_path "$BASELINE_OUT" --save_csv baseline_metrics.csv

echo "Evaluating U-ViT outputs"
python benchmark/evaluate_pie_bench.py --source_path "$PIE_ROOT" --output_path "$UVIT_OUT" --save_csv uvit_metrics.csv

echo "Comparison summary (means)"
python - <<'PY'
import pandas as pd
try:
    b=pd.read_csv('baseline_metrics.csv')
    u=pd.read_csv('uvit_metrics.csv')
    keys=['clip_whole','clip_edited','lpips_x1000','ssim_x100','psnr_raw']
    for k in keys:
        print(k, 'baseline', b[k].mean(), 'uvit', u[k].mean(), 'delta', u[k].mean()-b[k].mean())
except Exception as e:
    print('Error comparing CSVs:', e)
PY

echo "Pipeline complete. Baseline metrics saved to baseline_metrics.csv; UViT metrics saved to uvit_metrics.csv"
