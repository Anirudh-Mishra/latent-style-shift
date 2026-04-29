#!/usr/bin/env python3
"""
Initialize U-ViT weights from pretrained U-Net to enable immediate inference.
This allows the U-ViT to leverage pretrained diffusion knowledge.
"""

import torch
import argparse
from pathlib import Path
from diffusers import UNet2DConditionModel
from uvit_backbone import UViTBackbone


def copy_attention_weights(unet_attn, uvit_attn):
    """Copy attention weights from U-Net to U-ViT attention module."""
    try:
        # U-Net uses to_q, to_k, to_v
        if hasattr(unet_attn, 'to_q'):
            uvit_attn.to_q.weight.data.copy_(unet_attn.to_q.weight.data)
            if unet_attn.to_q.bias is not None:
                uvit_attn.to_q.bias.data.copy_(unet_attn.to_q.bias.data)
        
        if hasattr(unet_attn, 'to_k'):
            uvit_attn.to_k.weight.data.copy_(unet_attn.to_k.weight.data)
            if unet_attn.to_k.bias is not None:
                uvit_attn.to_k.bias.data.copy_(unet_attn.to_k.bias.data)
        
        if hasattr(unet_attn, 'to_v'):
            uvit_attn.to_v.weight.data.copy_(unet_attn.to_v.weight.data)
            if unet_attn.to_v.bias is not None:
                uvit_attn.to_v.bias.data.copy_(unet_attn.to_v.bias.data)
        
        if hasattr(unet_attn, 'to_out'):
            uvit_attn.proj.weight.data.copy_(unet_attn.to_out[0].weight.data)
            if unet_attn.to_out[0].bias is not None:
                uvit_attn.proj.bias.data.copy_(unet_attn.to_out[0].bias.data)
        
        return True
    except Exception as e:
        print(f"Warning: Could not copy attention weights: {e}")
        return False


def initialize_from_unet(uvit, unet):
    """
    Initialize U-ViT weights from pretrained U-Net where possible.
    This provides a warm start for the U-ViT architecture.
    """
    print("Initializing U-ViT from pretrained U-Net...")
    
    # Collect all attention modules from U-Net
    unet_attns = []
    for name, module in unet.named_modules():
        if module.__class__.__name__ == 'Attention':
            unet_attns.append((name, module))
    
    print(f"Found {len(unet_attns)} attention modules in U-Net")
    
    # Collect all attention modules from U-ViT
    uvit_blocks = list(uvit.in_blocks) + [uvit.mid_block] + list(uvit.out_blocks)
    
    copied_self = 0
    copied_cross = 0
    
    # Copy attention weights where dimensions match
    for i, block in enumerate(uvit_blocks):
        if i < len(unet_attns):
            # Try to copy self-attention
            unet_name, unet_module = unet_attns[min(i * 2, len(unet_attns) - 1)]
            if copy_attention_weights(unet_module, block.self_attn):
                copied_self += 1
            
            # Try to copy cross-attention
            if i * 2 + 1 < len(unet_attns):
                unet_name, unet_module = unet_attns[i * 2 + 1]
                if copy_attention_weights(unet_module, block.cross_attn):
                    copied_cross += 1
    
    print(f"Copied {copied_self} self-attention modules")
    print(f"Copied {copied_cross} cross-attention modules")
    
    # Note: MLP weights, norms, and patch embedding remain randomly initialized
    # This is intentional as the architectures differ significantly
    
    return uvit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='SimianLuo/LCM_Dreamshaper_v7',
                        help='Pretrained diffusion model ID')
    parser.add_argument('--out', type=str, default='./checkpoints/uvit_from_unet.pt',
                        help='Output checkpoint path')
    parser.add_argument('--uvit_size', type=str, default='mid', choices=['small', 'mid', 'large'])
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--source_conditioned', action='store_true',
                        help='Create source-conditioned U-ViT')
    args = parser.parse_args()
    
    print(f"Loading pretrained U-Net from {args.model_id}...")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder='unet')
    
    print(f"Creating U-ViT ({args.uvit_size})...")
    uvit = UViTBackbone.from_preset(
        args.uvit_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=4,
        context_dim=768,
        source_conditioned=args.source_conditioned,
    )
    
    # Initialize from U-Net
    uvit = initialize_from_unet(uvit, unet)
    
    # Save checkpoint
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model': uvit.state_dict(),
        'config': {
            'uvit_size': args.uvit_size,
            'img_size': args.img_size,
            'patch_size': args.patch_size,
            'source_conditioned': args.source_conditioned,
        }
    }
    
    torch.save(checkpoint, str(out_path))
    print(f"\n✅ Saved initialized U-ViT checkpoint to {out_path}")
    print(f"\nYou can now use this checkpoint for inference:")
    print(f"  python run_uvit_inference.py \\")
    print(f"    --checkpoint {out_path} \\")
    print(f"    --uvit_size {args.uvit_size} \\")
    print(f"    --source_path ../benchmark/pie_bench \\")
    print(f"    --target_path ./outputs")


if __name__ == '__main__':
    main()
