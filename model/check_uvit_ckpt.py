import argparse
import torch
import os

from uvit_backbone import UViTBackbone


def inspect_ckpt(path):
    sd = torch.load(path, map_location='cpu')
    total = 0
    zeros = 0
    print('Checkpoint:', path)
    for k,v in sd.items():
        if not isinstance(v, torch.Tensor):
            try:
                v = torch.tensor(v)
            except Exception:
                print(k, 'non-tensor ->', type(v))
                continue
        total += v.numel()
        zeros += (v == 0).sum().item()
        print(f"{k}: mean={v.mean().item():.6g}, std={v.std().item():.6g}, zeros={(v==0).sum().item()}")
    print('total params:', total, 'zeros:', zeros)


def forward_check(path, preset='mid'):
    sd = torch.load(path, map_location='cpu')
    bb = UViTBackbone.from_preset(preset, img_size=64, patch_size=2, in_chans=4)
    bb.load_state_dict(sd, strict=False)
    bb.eval()
    import torch
    sample = torch.randn(1,4,64,64)
    timesteps = torch.tensor([10])
    enc = torch.randn(1,77,768)
    out = bb(sample, timesteps, enc)
    print('input mean,std', sample.mean().item(), sample.std().item())
    print('output mean,std', out.mean().item(), out.std().item())
    print('max abs diff', (sample - out).abs().max().item())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--forward', action='store_true')
    p.add_argument('--preset', default='mid')
    args = p.parse_args()
    if not os.path.exists(args.ckpt):
        print('Checkpoint not found:', args.ckpt)
    else:
        inspect_ckpt(args.ckpt)
        if args.forward:
            forward_check(args.ckpt, preset=args.preset)
