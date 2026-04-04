import torch
import torch.nn as nn
from types import SimpleNamespace
from uvit_backbone import UViTBackbone, UVIT_CONFIGS


class UViTOutput:
    def __init__(self, sample):
        self.sample = sample


class UViTNamedBlock(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks if isinstance(blocks, nn.ModuleList) else nn.ModuleList([blocks])


class UViTAdapter(nn.Module):
    def __init__(self, backbone: UViTBackbone):
        super().__init__()
        self.backbone = backbone

        self.down = UViTNamedBlock(backbone.in_blocks)
        self.mid = UViTNamedBlock(backbone.mid_block)
        self.up = UViTNamedBlock(backbone.out_blocks)

        self.config = SimpleNamespace(
            _diffusers_version="0.999.0",
            sample_size=backbone.img_size,
            block_out_channels=[backbone.embed_dim],
            in_channels=backbone.in_chans,
        )

        self._dtype = None

    @property
    def dtype(self):
        if self._dtype is not None:
            return self._dtype
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    def __call__(self, sample, timestep, encoder_hidden_states=None,
                 cross_attention_kwargs=None, **kwargs):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(sample.shape[0])

        noise_pred = self.backbone(sample, timestep, encoder_hidden_states)
        return UViTOutput(noise_pred)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self._dtype = next(self.parameters()).dtype
        except StopIteration:
            pass
        return result


def create_uvit_adapter(preset="mid", **overrides):
    backbone = UViTBackbone.from_preset(preset, **overrides)
    return UViTAdapter(backbone)
