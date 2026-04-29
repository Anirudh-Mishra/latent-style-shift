#!/usr/bin/env python3
"""Diagnose checkpoint configuration issues."""

import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python diagnose_checkpoint.py <checkpoint.pt>")
    sys.exit(1)

ckpt_path = sys.argv[1]
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')

state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt))

print("\n=== CHECKPOINT ANALYSIS ===\n")

# 1. Check embedding dimension
if 'in_blocks.0.norm1.weight' in state_dict:
    embed_dim = state_dict['in_blocks.0.norm1.weight'].shape[0]
    print(f"✓ embed_dim: {embed_dim}")
else:
    print("✗ Could not find embed_dim")

# 2. Check depth
num_in = sum(1 for k in state_dict.keys() if k.startswith('in_blocks.') and '.norm1.weight' in k)
num_out = sum(1 for k in state_dict.keys() if k.startswith('out_blocks.') and '.norm1.weight' in k)
has_mid = 'mid_block.norm1.weight' in state_dict
depth = num_in + (1 if has_mid else 0) + num_out
print(f"✓ depth: {depth} (in={num_in}, mid={1 if has_mid else 0}, out={num_out})")

# 3. Check patch embedding
if 'patch_embed.proj.weight' in state_dict:
    patch_weight = state_dict['patch_embed.proj.weight']
    in_chans = patch_weight.shape[1]
    out_dim = patch_weight.shape[0]
    kernel_h, kernel_w = patch_weight.shape[2], patch_weight.shape[3]
    print(f"✓ patch_embed: in_chans={in_chans}, out_dim={out_dim}, kernel={kernel_h}x{kernel_w}")
    print(f"  → patch_size={kernel_h} (assuming square patches)")
else:
    print("✗ Could not find patch_embed")

# 4. Check positional embedding
if 'pos_embed' in state_dict:
    pos_embed = state_dict['pos_embed']
    print(f"✓ pos_embed: shape={pos_embed.shape}")
    num_patches = pos_embed.shape[1] - 1  # subtract time token
    print(f"  → num_patches={num_patches}")
    
    # Try to infer img_size
    if 'patch_embed.proj.weight' in state_dict:
        patch_size = state_dict['patch_embed.proj.weight'].shape[2]
        # num_patches = (img_size / patch_size)^2
        import math
        img_size = int(math.sqrt(num_patches) * patch_size)
        print(f"  → inferred img_size={img_size} (from {num_patches} patches with patch_size={patch_size})")
else:
    print("✗ Could not find pos_embed")

# 5. Check decoder
if 'decoder_pred.weight' in state_dict:
    decoder_weight = state_dict['decoder_pred.weight']
    out_chans = decoder_weight.shape[0]
    in_dim = decoder_weight.shape[1]
    print(f"✓ decoder_pred: in_dim={in_dim}, out_chans={out_chans}")
else:
    print("✗ Could not find decoder_pred")

# 6. Summary
print("\n=== RECOMMENDED SETTINGS ===\n")
if 'in_blocks.0.norm1.weight' in state_dict:
    embed_dim = state_dict['in_blocks.0.norm1.weight'].shape[0]
    if embed_dim == 512:
        print("--uvit_size small")
    elif embed_dim == 768:
        print("--uvit_size mid  # BUT embed_dim=768 (ViT-Base)")
    elif embed_dim == 1024:
        print("--uvit_size large")
    
if 'patch_embed.proj.weight' in state_dict:
    patch_size = state_dict['patch_embed.proj.weight'].shape[2]
    in_chans = state_dict['patch_embed.proj.weight'].shape[1]
    print(f"--patch_size {patch_size}")
    print(f"--in_chans {in_chans}")
    
if 'pos_embed' in state_dict and 'patch_embed.proj.weight' in state_dict:
    pos_embed = state_dict['pos_embed']
    num_patches = pos_embed.shape[1] - 1
    patch_size = state_dict['patch_embed.proj.weight'].shape[2]
    import math
    img_size = int(math.sqrt(num_patches) * patch_size)
    print(f"--img_size {img_size}")

print("\n=== POTENTIAL ISSUES ===\n")

# Check for mismatches
if 'in_blocks.0.norm1.weight' in state_dict:
    embed_dim = state_dict['in_blocks.0.norm1.weight'].shape[0]
    if embed_dim == 768:
        print("⚠️  WARNING: embed_dim=768 but 'mid' preset uses 512!")
        print("   This will cause size mismatch errors.")
        print("   FIX: Pass embed_dim=768 as override when creating model")

if 'pos_embed' in state_dict and 'patch_embed.proj.weight' in state_dict:
    pos_embed = state_dict['pos_embed']
    num_patches = pos_embed.shape[1] - 1
    patch_size = state_dict['patch_embed.proj.weight'].shape[2]
    import math
    img_size = int(math.sqrt(num_patches) * patch_size)
    
    # Latent size should be 64 for 512x512 images
    if img_size != 64:
        print(f"⚠️  WARNING: img_size={img_size} but latent space is 64x64!")
        print("   This will cause grid artifacts.")
        print("   FIX: Reinitialize checkpoint with --img_size 64")
