import os
import sys
import torch

# ensure repo model/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from uvit_adapter import UViTAdapter
from uvit_backbone import UViTBackbone


def main():
    torch.manual_seed(0)
    # small dummy model to keep compute small
    backbone = UViTBackbone.from_preset('small', img_size=64, patch_size=2, in_chans=4)
    adapter = UViTAdapter(backbone)

    B = 2
    C = 4
    H = W = 64
    sample = torch.randn(B, C, H, W)
    timesteps = torch.randint(0, 1000, (B,), dtype=torch.long)
    encoder_hidden_states = torch.randn(B, 77, 768)

    out = adapter(sample, timesteps, encoder_hidden_states).sample

    print('input shape:', sample.shape)
    print('output shape:', out.shape)
    diff = (sample - out).abs().max().item()
    print('max absolute difference:', diff)
    print('equal (allclose):', torch.allclose(sample, out, atol=1e-6))

    if diff <= 1e-6:
        print('IDENTICAL: output equals input (within tol)')
        sys.exit(1)
    else:
        print('OK: output differs from input')
        sys.exit(0)


if __name__ == '__main__':
    main()
