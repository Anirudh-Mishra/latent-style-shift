import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
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

    def forward(self, sample, timestep, encoder_hidden_states=None,
                 cross_attention_kwargs=None, source_latent=None, **kwargs):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(sample.shape[0])

        noise_pred = self.backbone(sample, timestep, encoder_hidden_states,
                                   source_latent=source_latent)
        return UViTOutput(noise_pred)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self._dtype = next(self.parameters()).dtype
        except StopIteration:
            pass
        return result


def register_attention_control(adapter: UViTAdapter, controller):
    """
    Register the UAC controller on every self-attention and cross-attention
    module in the UViT backbone.

    - in_blocks  → place_in_unet = "down"
    - mid_block  → place_in_unet = "mid"
    - out_blocks → place_in_unet = "up"

    Processors follow the module-level convention used by ptp_utils:
      SelfAttention.processor(attn_module, x)          -> output (B, L, C)
      CrossAttention.processor(attn_module, x, context) -> output (B, L, C)
    """
    backbone = adapter.backbone
    num_att_layers = 0

    def _make_self_processor(ctrl, place):
        def processor(attn_module, x):
            B, L, C = x.shape
            h = attn_module.num_heads
            D = attn_module.head_dim
            # Project and reshape to (B*h, L, D)
            q = attn_module.to_q(x).reshape(B, L, h, D).permute(0, 2, 1, 3).reshape(B * h, L, D)
            k = attn_module.to_k(x).reshape(B, L, h, D).permute(0, 2, 1, 3).reshape(B * h, L, D)
            v = attn_module.to_v(x).reshape(B, L, h, D).permute(0, 2, 1, 3).reshape(B * h, L, D)

            # UAC self-attention sharing — may change the batch dimension
            q, k, v = ctrl.self_attn_forward(q, k, v, h)

            # Derive actual batch size AFTER UAC — do NOT reuse original B
            # self_attn_forward can change stream count (e.g. 2 streams -> 2 streams
            # but with source k/v injected, or rearranged row order)
            B_out = q.shape[0] // h

            # Use Flash Attention via SDPA; reshape to (B_out, h, L, D) format.
            # Cross-attention processor keeps manual attn because UAC needs the map.
            out = F.scaled_dot_product_attention(
                q.view(B_out, h, L, D),
                k.view(B_out, h, L, D),
                v.view(B_out, h, L, D),
                dropout_p=attn_module.attn_drop.p if attn_module.training else 0.0,
            )  # (B_out, h, L, D)
            out = out.permute(0, 2, 1, 3).reshape(B_out, L, h * D)
            out = attn_module.proj(out)
            out = attn_module.proj_drop(out)
            return out
        return processor

    def _make_cross_processor(ctrl, place):
        def processor(attn_module, x, context):
            B, L, C = x.shape
            h = attn_module.num_heads
            D = attn_module.head_dim
            Lk = context.shape[1]

            q = attn_module.to_q(x).reshape(B, L, h, D).permute(0, 2, 1, 3).reshape(B * h, L, D)
            k = attn_module.to_k(context).reshape(B, Lk, h, D).permute(0, 2, 1, 3).reshape(B * h, Lk, D)
            v = attn_module.to_v(context).reshape(B, Lk, h, D).permute(0, 2, 1, 3).reshape(B * h, Lk, D)

            attn = (q @ k.transpose(-2, -1)) * attn_module.scale
            attn = attn.softmax(dim=-1)

            # UAC cross-attention control — controller observes/modifies attn map
            attn = ctrl(attn, is_cross=True, place_in_unet=place)
            attn = attn_module.attn_drop(attn)

            # Re-derive B_out after controller may have changed attn batch dim
            B_out = attn.shape[0] // h

            out = attn @ v  # (B_out*h, L, D)
            out = out.reshape(B_out, h, L, D).permute(0, 2, 1, 3).reshape(B_out, L, h * D)
            out = attn_module.proj(out)
            out = attn_module.proj_drop(out)
            return out
        return processor

    def _register_block(block, place: str):
        nonlocal num_att_layers
        block.self_attn._place_in_unet = place
        block.cross_attn._place_in_unet = place
        block.self_attn.processor = _make_self_processor(controller, place)
        block.cross_attn.processor = _make_cross_processor(controller, place)
        num_att_layers += 2  # one self + one cross per block

    for blk in backbone.in_blocks:
        _register_block(blk, "down")
    _register_block(backbone.mid_block, "mid")
    for blk in backbone.out_blocks:
        _register_block(blk, "up")

    controller.num_att_layers = num_att_layers


def unregister_attention_control(adapter: UViTAdapter):
    """Remove all UAC processors from the backbone (restore default forward)."""
    backbone = adapter.backbone
    all_blocks = list(backbone.in_blocks) + [backbone.mid_block] + list(backbone.out_blocks)
    for blk in all_blocks:
        blk.self_attn.processor = None
        blk.cross_attn.processor = None


def has_attention_processors(adapter: UViTAdapter) -> bool:
    """Return True iff UAC processors are currently registered."""
    backbone = adapter.backbone
    all_blocks = list(backbone.in_blocks) + [backbone.mid_block] + list(backbone.out_blocks)
    return any(
        blk.self_attn.processor is not None or blk.cross_attn.processor is not None
        for blk in all_blocks
    )


def create_uvit_adapter(preset="mid", **overrides):
    backbone = UViTBackbone.from_preset(preset, **overrides)
    return UViTAdapter(backbone)