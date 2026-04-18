import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from uvit_backbone import UViTBackbone


def resize_pos_embed(src, dst_num_patches, embed_dim):
    # src: (1, N+1, C) or (1, N, C)
    src = src.float()
    if src.ndim != 3:
        raise ValueError('unexpected pos_embed shape')

    N = src.shape[1]
    # detect and remove cls token when present (common case: N = 1 + S*S)
    s1 = int(math.sqrt(N))
    if s1 * s1 == N:
        src_tok = src
    else:
        # try removing first token
        if (N - 1) > 0 and int(math.sqrt(N - 1)) ** 2 == (N - 1):
            src_tok = src[:, 1:, :]
        else:
            # fallback: use as-is
            src_tok = src
    N = src_tok.shape[1]
    S = int(math.sqrt(N))
    if S * S != N:
        raise ValueError(f'pos_embed token count {N} is not a square')
    D = int(math.sqrt(dst_num_patches))
    src_grid = src_tok.reshape(1, S, S, embed_dim).permute(0, 3, 1, 2)  # (1,C,S,S)
    D = int(math.sqrt(dst_num_patches))
    src_grid = src_tok.reshape(1, S, S, embed_dim).permute(0, 3, 1, 2)  # (1,C,S,S)
    # interpolate to (1,C,D,D)
    resized = F.interpolate(src_grid, size=(D, D), mode='bicubic', align_corners=False)
    resized = resized.permute(0, 2, 3, 1).reshape(1, D * D, embed_dim)
    return resized


def map_block_from_mae(src, dst):
    # src: dict for one MAE block
    # dst: UViT TransformerBlock
    # map norms
    if f"norm1.weight" in src:
        dst.norm1.weight.data.copy_(src[f"norm1.weight"]) if hasattr(dst.norm1, 'weight') else None
        dst.norm1.bias.data.copy_(src[f"norm1.bias"]) if hasattr(dst.norm1, 'bias') else None

    # attention qkv -> split
    if f"attn.qkv.weight" in src:
        qkv_w = src[f"attn.qkv.weight"]
        C = qkv_w.shape[1]
        chunk = qkv_w.shape[0] // 3
        q_w, k_w, v_w = qkv_w.chunk(3, 0)
        dst.self_attn.to_q.weight.data.copy_(q_w)
        dst.self_attn.to_k.weight.data.copy_(k_w)
        dst.self_attn.to_v.weight.data.copy_(v_w)
    if f"attn.qkv.bias" in src and dst.self_attn.to_q.bias is not None:
        qkv_b = src[f"attn.qkv.bias"]
        q_b, k_b, v_b = qkv_b.chunk(3, 0)
        dst.self_attn.to_q.bias.data.copy_(q_b)
        dst.self_attn.to_k.bias.data.copy_(k_b)
        dst.self_attn.to_v.bias.data.copy_(v_b)

    # attn proj
    if f"attn.proj.weight" in src:
        dst.self_attn.proj.weight.data.copy_(src[f"attn.proj.weight"]) 
    if f"attn.proj.bias" in src and dst.self_attn.proj.bias is not None:
        dst.self_attn.proj.bias.data.copy_(src[f"attn.proj.bias"]) 

    # norm2 -> map to norm2 and norm3 (UViT has 3 norms)
    if f"norm2.weight" in src:
        dst.norm2.weight.data.copy_(src[f"norm2.weight"]) if hasattr(dst.norm2, 'weight') else None
        dst.norm2.bias.data.copy_(src[f"norm2.bias"]) if hasattr(dst.norm2, 'bias') else None
        # also copy to norm3 when available
        if hasattr(dst, 'norm3'):
            dst.norm3.weight.data.copy_(src[f"norm2.weight"]) 
            dst.norm3.bias.data.copy_(src[f"norm2.bias"]) 

    # mlp
    if f"mlp.fc1.weight" in src:
        dst.mlp.fc1.weight.data.copy_(src[f"mlp.fc1.weight"]) 
    if f"mlp.fc1.bias" in src:
        dst.mlp.fc1.bias.data.copy_(src[f"mlp.fc1.bias"]) 
    if f"mlp.fc2.weight" in src:
        dst.mlp.fc2.weight.data.copy_(src[f"mlp.fc2.weight"]) 
    if f"mlp.fc2.bias" in src:
        dst.mlp.fc2.bias.data.copy_(src[f"mlp.fc2.bias"]) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-checkpoint', type=str, required=True)
    parser.add_argument('--out', type=str, default='model/checkpoints/uvit_init_from_mae.pth')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--in_chans', type=int, default=4)
    args = parser.parse_args()

    src_path = Path(args.source_checkpoint)
    assert src_path.exists(), f"source checkpoint {src_path} not found"

    ck = torch.load(str(src_path), map_location='cpu')
    # some checkpoints wrap in { 'model': OrderedDict }
    if isinstance(ck, dict) and 'model' in ck and isinstance(ck['model'], dict):
        src = ck['model']
    elif isinstance(ck, dict):
        src = ck
    else:
        raise RuntimeError('Unsupported checkpoint format')

    # infer embed dim and number of blocks
    if 'patch_embed.proj.weight' in src:
        embed_dim = src['patch_embed.proj.weight'].shape[0]
    elif 'pos_embed' in src:
        embed_dim = src['pos_embed'].shape[-1]
    else:
        raise RuntimeError('Cannot infer embed dim from checkpoint')

    # collect MAE blocks count
    block_keys = [k for k in src.keys() if k.startswith('blocks.') and k.endswith('norm1.weight')]
    num_mae_blocks = len(block_keys)
    print(f'Found embed_dim={embed_dim}, num_mae_blocks={num_mae_blocks}')

    # choose uvit depth: prefer odd depth close to num_mae_blocks
    if num_mae_blocks % 2 == 1:
        uvit_depth = num_mae_blocks
    else:
        uvit_depth = num_mae_blocks - 1
    print(f'Creating UViT with embed_dim={embed_dim}, depth={uvit_depth}')

    uvit = UViTBackbone(img_size=args.img_size, patch_size=args.patch_size, in_chans=args.in_chans,
                        embed_dim=embed_dim, depth=uvit_depth)

    # map positional embedding
    if 'pos_embed' in src:
        dst_num = uvit.num_patches
        resized = resize_pos_embed(src['pos_embed'], dst_num, embed_dim)
        # keep extras token (time token) in uvit.pos_embed; insert resized into[:,1:]
        with torch.no_grad():
            if uvit.pos_embed.shape[1] == resized.shape[1] + 1:
                uvit.pos_embed[:, 1:, :].copy_(resized)
            elif uvit.pos_embed.shape[1] == resized.shape[1]:
                uvit.pos_embed[:, :resized.shape[1], :].copy_(resized)
            else:
                print('pos_embed sizes differ, copying available entries')
                n = min(uvit.pos_embed.shape[1] - 1, resized.shape[1])
                uvit.pos_embed[:, 1:1 + n, :].copy_(resized[:, :n, :])

    # build target block list
    tgt_blocks = list(uvit.in_blocks) + [uvit.mid_block] + list(uvit.out_blocks)
    # map each available MAE block into target sequentially
    for i in range(min(len(tgt_blocks), num_mae_blocks)):
        prefix = f'blocks.{i}.'
        src_block = {k[len(prefix):]: v for k, v in src.items() if k.startswith(prefix)}
        print(f'Mapping MAE block {i} -> UViT block {i}')
        map_block_from_mae(src_block, tgt_blocks[i])

    # map final norm if present
    if 'norm.weight' in src:
        try:
            uvit.norm.weight.data.copy_(src['norm.weight'])
            uvit.norm.bias.data.copy_(src['norm.bias'])
        except Exception:
            pass

    # Save state dict
    out_path = Path(args.out)
    torch.save({'model_state_dict': uvit.state_dict()}, str(out_path))
    print(f'Saved mapped UViT checkpoint to {out_path}')


if __name__ == '__main__':
    main()
