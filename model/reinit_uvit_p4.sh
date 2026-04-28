#!/bin/bash
#
# Reinitialize U-ViT with patch_size=4 (larger patches = fewer boundaries)
#

echo "=========================================="
echo "Reinitializing U-ViT with patch_size=4"
echo "=========================================="

# Initialize from MAE with patch_size=4
python init_from_pretrained_vit.py \
  --source-checkpoint ./checkpoints/mae_pretrain_vit_base.pth \
  --out ./checkpoints/uvit_from_mae_p4.pt \
  --img_size 64 \
  --patch_size 4 \
  --in_chans 4

echo ""
echo "✅ Created checkpoint: ./checkpoints/uvit_from_mae_p4.pt"
echo ""
echo "Now train with:"
echo "  --patch_size 4"
echo "  --resume ./checkpoints/uvit_from_mae_p4.pt"
