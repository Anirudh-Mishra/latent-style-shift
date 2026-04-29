"""
Initialize UViT with hybrid pretraining:
  - Self-attention, MLPs, norms, patch embedding, pos_embed: from MAE-init UViT checkpoint
  - Cross-attention only: from LCM Dreamshaper UNet, resized via bilinear interpolation
    to match UViT's embed_dim.

Why this exists: MAE has no text encoder, so cross-attention layers in MAE-init UViT
checkpoints are random. UNet has billions of pairs of text-pixel co-supervision baked
into its cross-attention. Copying ONLY the cross-attention from UNet (and keeping
MAE's better self-attention) gives the UViT a warm start on text grounding while
preserving its image-structure prior.

The dimension mismatch between SD 1.5 UNet (320/640/1280-dim cross-attn) and UViT
(512-dim for the 'mid' preset) is bridged with bilinear interpolation on the weight
matrices. This is lossy — text grounding is approximately preserved, not perfectly
transferred — but is much better than random init.

Usage:
    python init_from_unet_cross_only.py \\
        --mae_checkpoint ./checkpoints/uvit_from_mae.pt \\
        --out ./checkpoints/uvit_mae_self_unet_cross.pt \\
        --uvit_size mid
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel

from uvit_backbone import UViTBackbone


def resize_2d_weight(weight: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
    """Resize a Linear weight matrix (out_src, in_src) to (out_dim, in_dim) via bilinear."""
    if weight.shape == (out_dim, in_dim):
        return weight.clone()
    w = weight.unsqueeze(0).unsqueeze(0).float()  # (1, 1, out_src, in_src)
    w = F.interpolate(w, size=(out_dim, in_dim), mode="bilinear", align_corners=False)
    return w.squeeze(0).squeeze(0)


def resize_1d_bias(bias: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Resize a Linear bias vector (out_src,) to (out_dim,) via 1D linear interpolation."""
    if bias.shape == (out_dim,):
        return bias.clone()
    b = bias.unsqueeze(0).unsqueeze(0).float()  # (1, 1, out_src)
    b = F.interpolate(b, size=out_dim, mode="linear", align_corners=False)
    return b.squeeze(0).squeeze(0)


def copy_resized_cross_attn(unet_attn, uvit_attn, ctx_dim: int = 768):
    """Copy & resize a UNet cross-attention's weights into a UViT cross-attention module.

    UNet cross-attn:  to_q (D_unet, D_unet),  to_k (D_unet, ctx),  to_v (D_unet, ctx),
                      to_out[0] (D_unet, D_unet)
    UViT cross-attn:  to_q (D_uvit, D_uvit),  to_k (D_uvit, ctx),  to_v (D_uvit, ctx),
                      proj  (D_uvit, D_uvit)
    Only D_unet (output dim) changes; CLIP context dim is 768 in both.
    """
    uvit_dim = uvit_attn.to_q.weight.shape[0]

    # to_q: resize both axes (D_unet, D_unet) -> (D_uvit, D_uvit)
    uvit_attn.to_q.weight.data.copy_(
        resize_2d_weight(unet_attn.to_q.weight.data, uvit_dim, uvit_dim)
    )
    if unet_attn.to_q.bias is not None and uvit_attn.to_q.bias is not None:
        uvit_attn.to_q.bias.data.copy_(resize_1d_bias(unet_attn.to_q.bias.data, uvit_dim))

    # to_k, to_v: resize output axis only, input axis is 768 in both
    uvit_attn.to_k.weight.data.copy_(
        resize_2d_weight(unet_attn.to_k.weight.data, uvit_dim, ctx_dim)
    )
    if unet_attn.to_k.bias is not None and uvit_attn.to_k.bias is not None:
        uvit_attn.to_k.bias.data.copy_(resize_1d_bias(unet_attn.to_k.bias.data, uvit_dim))

    uvit_attn.to_v.weight.data.copy_(
        resize_2d_weight(unet_attn.to_v.weight.data, uvit_dim, ctx_dim)
    )
    if unet_attn.to_v.bias is not None and uvit_attn.to_v.bias is not None:
        uvit_attn.to_v.bias.data.copy_(resize_1d_bias(unet_attn.to_v.bias.data, uvit_dim))

    # proj: UNet's output projection is to_out[0]. Note this overrides UViT's zero-init
    # of cross_attn.proj — that zero-init was a defensive choice for random
    # cross-attn; with UNet's pretrained values we want to use them directly.
    unet_proj_w = unet_attn.to_out[0].weight.data
    uvit_attn.proj.weight.data.copy_(resize_2d_weight(unet_proj_w, uvit_dim, uvit_dim))
    if unet_attn.to_out[0].bias is not None and uvit_attn.proj.bias is not None:
        uvit_attn.proj.bias.data.copy_(
            resize_1d_bias(unet_attn.to_out[0].bias.data, uvit_dim)
        )


def collect_unet_cross_attns(unet):
    """Collect cross-attention modules from UNet, grouped by encoder/mid/decoder.

    SD 1.5 UNet convention: in each transformer block, attn1 is self-attn and attn2
    is cross-attn. We filter to attn2 only.
    """
    down_attns, mid_attns, up_attns = [], [], []
    for name, module in unet.named_modules():
        if module.__class__.__name__ != "Attention":
            continue
        if not name.endswith(".attn2"):
            continue
        if "down_blocks" in name:
            down_attns.append((name, module))
        elif "mid_block" in name:
            mid_attns.append((name, module))
        elif "up_blocks" in name:
            up_attns.append((name, module))
    return down_attns, mid_attns, up_attns


def sample_evenly(src_len: int, tgt_len: int):
    """Pick tgt_len indices evenly spaced over [0, src_len-1]."""
    if src_len == 0 or tgt_len == 0:
        return []
    if tgt_len == 1:
        return [src_len // 2]
    return [int(round(i * (src_len - 1) / (tgt_len - 1))) for i in range(tgt_len)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mae_checkpoint", type=str, required=True,
                        help="Path to MAE-initialized UViT checkpoint (e.g. uvit_from_mae.pt).")
    parser.add_argument("--model_id", type=str, default="SimianLuo/LCM_Dreamshaper_v7",
                        help="HuggingFace ID of the UNet to source cross-attention from.")
    parser.add_argument("--out", type=str,
                        default="./checkpoints/uvit_mae_self_unet_cross.pt")
    parser.add_argument("--uvit_size", type=str, default="mid",
                        choices=["small", "mid", "large"])
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--source_conditioned", action="store_true",
                        help="Must match the flag used at training time.")
    args = parser.parse_args()

    # 1. Load MAE-init UViT checkpoint state dict
    print(f"Loading MAE-init UViT from {args.mae_checkpoint}...")
    mae_ck = torch.load(args.mae_checkpoint, map_location="cpu")
    if isinstance(mae_ck, dict) and "model" in mae_ck:
        mae_state = mae_ck["model"]
    else:
        mae_state = mae_ck
    # Strip torch.compile prefix if present
    if any(k.startswith("_orig_mod.") for k in mae_state):
        mae_state = {k.replace("_orig_mod.", "", 1): v for k, v in mae_state.items()}

    # Detect dimensions from the checkpoint so we instantiate UViT with matching shape
    embed_dim = mae_state["in_blocks.0.norm1.weight"].shape[0]
    num_in = sum(1 for k in mae_state if k.startswith("in_blocks.") and ".norm1.weight" in k)
    num_out = sum(1 for k in mae_state if k.startswith("out_blocks.") and ".norm1.weight" in k)
    depth = num_in + 1 + num_out
    print(f"Detected from checkpoint: embed_dim={embed_dim}, depth={depth}")

    # 2. Build UViT and load MAE state
    uvit = UViTBackbone.from_preset(
        args.uvit_size,
        embed_dim=embed_dim,
        depth=depth,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=4,
        context_dim=768,
        source_conditioned=args.source_conditioned,
    )
    missing, unexpected = uvit.load_state_dict(mae_state, strict=False)
    print(f"Loaded MAE state: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    if unexpected:
        print(f"  (unexpected examples: {unexpected[:3]})")

    # 3. Load UNet
    print(f"Loading UNet from {args.model_id}...")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    unet.eval()

    # 4. Find UNet cross-attention layers
    down_attns, mid_attns, up_attns = collect_unet_cross_attns(unet)
    print(f"Found UNet cross-attentions: {len(down_attns)} down, "
          f"{len(mid_attns)} mid, {len(up_attns)} up")
    if len(down_attns) == 0 or len(mid_attns) == 0 or len(up_attns) == 0:
        raise RuntimeError("UNet did not yield expected cross-attention layers. "
                           "Verify the model_id points to a CrossAttn UNet.")

    # 5. Map UNet cross-attns to UViT blocks (encoder/mid/decoder symmetry)
    n_in = len(uvit.in_blocks)
    n_out = len(uvit.out_blocks)
    in_idxs = sample_evenly(len(down_attns), n_in)
    out_idxs = sample_evenly(len(up_attns), n_out)

    print("Mapping plan:")
    print(f"  in_blocks ({n_in}):  UNet down indices {in_idxs}")
    print(f"  mid_block (1):  UNet mid index 0")
    print(f"  out_blocks ({n_out}): UNet up indices  {out_idxs}")

    # 6. Copy & resize cross-attentions
    n_copied = 0
    for i, src_idx in enumerate(in_idxs):
        _, src_attn = down_attns[src_idx]
        copy_resized_cross_attn(src_attn, uvit.in_blocks[i].cross_attn)
        n_copied += 1

    copy_resized_cross_attn(mid_attns[0][1], uvit.mid_block.cross_attn)
    n_copied += 1

    for i, src_idx in enumerate(out_idxs):
        _, src_attn = up_attns[src_idx]
        copy_resized_cross_attn(src_attn, uvit.out_blocks[i].cross_attn)
        n_copied += 1

    print(f"Copied & resized cross-attention into {n_copied} UViT blocks.")

    # 7. Sanity check: cross-attn weights should no longer be near-zero or near-random
    sample_proj = uvit.in_blocks[0].cross_attn.proj.weight.data
    print(f"Sanity check on in_blocks[0].cross_attn.proj.weight: "
          f"mean={sample_proj.mean().item():.4e}, "
          f"std={sample_proj.std().item():.4e}, "
          f"abs_max={sample_proj.abs().max().item():.4e}")
    if sample_proj.abs().max().item() < 1e-6:
        raise RuntimeError("Cross-attention proj is still ~zero after copy. "
                           "Something went wrong with the transfer.")

    # 8. Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": uvit.state_dict()}, str(out_path))
    print(f"\nSaved hybrid checkpoint to {out_path}")
    print("  Self-attention/MLPs/norms: from MAE")
    print("  Cross-attention: from UNet (resized via bilinear interpolation)")


if __name__ == "__main__":
    main()
