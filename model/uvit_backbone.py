import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


UVIT_CONFIGS = {
    "small": dict(embed_dim=256, depth=12, num_heads=4, mlp_ratio=4.0),
    "mid":   dict(embed_dim=512, depth=12, num_heads=8, mlp_ratio=4.0),
    "large": dict(embed_dim=768, depth=16, num_heads=12, mlp_ratio=4.0),
}


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    return einops.rearrange(
        imgs, "B C (h p1) (w p2) -> B (h w) (p1 p2 C)", p1=patch_size, p2=patch_size
    )


def unpatchify(x, channels, patch_size, h, w):
    return einops.rearrange(
        x,
        "B (h w) (p1 p2 C) -> B C (h p1) (w p2)",
        h=h, w=w, p1=patch_size, p2=patch_size, C=channels,
    )


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=4, embed_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.processor = None

    def _default_forward(self, x):
        B, L, C = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = einops.rearrange(q, "B L (H D) -> (B H) L D", H=self.num_heads)
        k = einops.rearrange(k, "B L (H D) -> (B H) L D", H=self.num_heads)
        v = einops.rearrange(v, "B L (H D) -> (B H) L D", H=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = einops.rearrange(out, "(B H) L D -> B L (H D)", H=self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def forward(self, x):
        if self.processor is not None:
            return self.processor(self, x)
        return self._default_forward(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=768, num_heads=8, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.processor = None

    def _default_forward(self, x, context):
        B, L, C = x.shape

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = einops.rearrange(q, "B L (H D) -> (B H) L D", H=self.num_heads)
        k = einops.rearrange(k, "B S (H D) -> (B H) S D", H=self.num_heads)
        v = einops.rearrange(v, "B S (H D) -> (B H) S D", H=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = einops.rearrange(out, "(B H) L D -> B L (H D)", H=self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def forward(self, x, context):
        if self.processor is not None:
            return self.processor(self, x, context)
        return self._default_forward(x, context)


class TransformerBlock(nn.Module):
    def __init__(self, dim, context_dim=768, num_heads=8, mlp_ratio=4.0,
                 qkv_bias=True, norm_layer=nn.LayerNorm, skip=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, context_dim=context_dim,
                                         num_heads=num_heads, qkv_bias=qkv_bias)

        self.norm3 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden)

        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

    def forward(self, x, context, skip=None):
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))

        return x


class UViTBackbone(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=2,
        in_chans=4,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        context_dim=768,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        conv_output=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        self.extras = 1
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.extras + self.num_patches, embed_dim)
        )

        self.in_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, context_dim=context_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                skip=False,
            )
            for _ in range(depth // 2)
        ])

        self.mid_block = TransformerBlock(
            dim=embed_dim, context_dim=context_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
            skip=False,
        )

        self.out_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, context_dim=context_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                skip=True,
            )
            for _ in range(depth // 2)
        ])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = (
            nn.Conv2d(in_chans, in_chans, 3, padding=1) if conv_output else nn.Identity()
        )

        self._init_pos_embed()
        self.apply(self._init_weights)

    def _init_pos_embed(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @classmethod
    def from_preset(cls, preset="mid", **overrides):
        if preset not in UVIT_CONFIGS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from {list(UVIT_CONFIGS.keys())}")
        cfg = {**UVIT_CONFIGS[preset], **overrides}
        return cls(**cfg)

    def forward(self, x, timesteps, encoder_hidden_states):
        B, C, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        x = self.patch_embed(x)

        # Generate the embeddings
        t_emb = timestep_embedding(timesteps, self.embed_dim)

        # Cast them to match the dtype of the incoming sample/latents
        t_emb = t_emb.to(dtype=x.dtype) 

        # Pass through the linear layers
        time_token = self.time_embed(t_emb)
        time_token = time_token.unsqueeze(1)

        x = torch.cat([time_token, x], dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x, context=encoder_hidden_states)
            skips.append(x)

        x = self.mid_block(x, context=encoder_hidden_states)

        for blk in self.out_blocks:
            x = blk(x, context=encoder_hidden_states, skip=skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)

        x = x[:, self.extras:, :]
        x = unpatchify(x, self.in_chans, self.patch_size, h_patches, w_patches)

        x = self.final_layer(x)

        return x
