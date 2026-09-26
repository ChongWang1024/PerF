import torch
from torch import nn

from util.model_util import (
    Attention,
    BottleneckPatchEmbed,
    LabelEmbedder,
    RMSNorm,
    SwiGLUFFN,
    TimestepEmbedder,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
)


class PToAConditioning(nn.Module):
    def __init__(self, persistent_width, active_width, rank):
        super().__init__()
        self.norm = RMSNorm(persistent_width, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.Linear(persistent_width, rank),
            nn.SiLU(),
            nn.Linear(rank, 6 * active_width),
        )

    def forward(self, persistent):
        return self.modulation(self.norm(persistent))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, cond_dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True),
        )

    @torch.compile(dynamic=False)
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class PerFBlock(nn.Module):
    def __init__(
        self,
        global_width,
        active_width,
        cond_dim=768,
        p_to_a_rank=128,
        head_dim=64,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if active_width > global_width:
            raise ValueError(
                f"active_width={active_width} cannot exceed global_width={global_width}"
            )
        if active_width % head_dim != 0:
            raise ValueError(
                f"active_width={active_width} must be divisible by head_dim={head_dim}"
            )

        self.global_width = int(global_width)
        self.active_width = int(active_width)
        self.cond_dim = int(cond_dim)
        self.p_to_a_rank = int(p_to_a_rank)
        self.persistent_width = self.global_width - self.active_width
        self.num_heads = self.active_width // head_dim

        self.norm1 = RMSNorm(self.active_width, eps=1e-6)
        self.attn = Attention(
            self.active_width,
            num_heads=self.num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(self.active_width, eps=1e-6)
        self.mlp = SwiGLUFFN(
            self.active_width, int(self.active_width * mlp_ratio), drop=proj_drop
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 6 * self.active_width, bias=True),
        )

        self.p_to_a = (
            PToAConditioning(self.persistent_width, active_width, p_to_a_rank)
            if self.persistent_width
            else None
        )

    @torch.compile(dynamic=False)
    def forward(self, x, c, feat_rope=None, p_to_a_mask=None, p_to_a_scale=1.0):
        x_active = x[..., : self.active_width].contiguous()
        x_persistent = x[..., self.active_width :].contiguous()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa = gate_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)

        if self.p_to_a is not None:
            persistent_mod = self.p_to_a(x_persistent)
            if p_to_a_mask is not None:
                persistent_mod = persistent_mod * p_to_a_mask.to(
                    dtype=persistent_mod.dtype
                )
            persistent_mod = persistent_mod * p_to_a_scale
            (
                persistent_shift_msa,
                persistent_scale_msa,
                persistent_gate_msa,
                persistent_shift_mlp,
                persistent_scale_mlp,
                persistent_gate_mlp,
            ) = persistent_mod.chunk(6, dim=-1)
            shift_msa = shift_msa + persistent_shift_msa
            scale_msa = scale_msa + persistent_scale_msa
            gate_msa = gate_msa + persistent_gate_msa
            shift_mlp = shift_mlp + persistent_shift_mlp
            scale_mlp = scale_mlp + persistent_scale_mlp
            gate_mlp = gate_mlp + persistent_gate_mlp

        x_active = x_active + gate_msa * self.attn(
            self.norm1(x_active) * (1 + scale_msa) + shift_msa,
            rope=feat_rope,
        )
        x_active = x_active + gate_mlp * self.mlp(
            self.norm2(x_active) * (1 + scale_mlp) + shift_mlp
        )

        if x_persistent.shape[-1] == 0:
            return x_active
        return torch.cat([x_active, x_persistent], dim=-1).contiguous()


class PerF(nn.Module):
    """
    Just image Transformer with a fixed global residual stream and
    per-layer active channel widths.
    """

    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=768,
        width_schedule=None,
        cond_dim=None,
        p_to_a_rank=128,
        p_to_a_dropout=0.1,
        head_dim=64,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
    ):
        super().__init__()
        if width_schedule is None:
            raise ValueError("width_schedule must be provided")
        if len(width_schedule) == 0:
            raise ValueError("width_schedule must contain at least one layer")

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.cond_dim = hidden_size if cond_dim is None else int(cond_dim)
        self.input_size = input_size
        self.in_context_len = int(in_context_len)
        self.in_context_start = (
            None if self.in_context_len <= 0 else int(in_context_start)
        )
        self.num_classes = num_classes
        self.head_dim = head_dim
        self.p_to_a_dropout = float(p_to_a_dropout)
        if not 0 <= self.p_to_a_dropout <= 1:
            raise ValueError("p_to_a_dropout must be in [0, 1]")
        if input_size % patch_size:
            raise ValueError("input_size must be divisible by patch_size")
        self.width_schedule = tuple(int(width) for width in width_schedule)
        self.active_heads = tuple(width // head_dim for width in self.width_schedule)
        if self.in_context_len < 0:
            raise ValueError(f"in_context_len must be >= 0, got {self.in_context_len}")
        if self.in_context_len > 0 and not 0 <= self.in_context_start < len(
            self.width_schedule
        ):
            raise ValueError(
                f"Expected 0 <= in_context_start < {len(self.width_schedule)} when in_context_len > 0, "
                f"got {self.in_context_start}"
            )

        for width in self.width_schedule:
            if width <= 0 or width > hidden_size:
                raise ValueError(
                    f"All active widths must be <= hidden_size={hidden_size}, got {width}"
                )
            if width % head_dim != 0:
                raise ValueError(
                    f"All active widths must be divisible by head_dim={head_dim}, got {width}"
                )

        self.t_embedder = TimestepEmbedder(self.cond_dim)
        self.y_embedder = LabelEmbedder(num_classes, self.cond_dim)
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )
        self.y_token_proj = (
            nn.Identity()
            if self.cond_dim == hidden_size
            else nn.Linear(self.cond_dim, hidden_size, bias=True)
        )

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True
            )
            torch.nn.init.normal_(self.in_context_posemb, std=0.02)
        else:
            self.in_context_posemb = None

        hw_seq_len = input_size // patch_size
        rope_dim = head_dim // 2
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=rope_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0,
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=rope_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len,
        )

        depth = len(self.width_schedule)
        self.blocks = nn.ModuleList(
            [
                PerFBlock(
                    hidden_size,
                    active_width,
                    cond_dim=self.cond_dim,
                    p_to_a_rank=p_to_a_rank,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                )
                for i, active_width in enumerate(self.width_schedule)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size, self.cond_dim, patch_size, self.out_channels
        )
        self.feature_layer = self.width_schedule.index(min(self.width_schedule))
        self.persistent_start = min(self.width_schedule)
        self.persistent_width = hidden_size - self.persistent_start

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            if block.p_to_a is not None:
                nn.init.zeros_(block.p_to_a.modulation[-1].weight)
                nn.init.zeros_(block.p_to_a.modulation[-1].bias)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, y, return_persistent=False, p_to_a_scale=1.0):
        t_emb, y_emb = self.t_embedder(t), self.y_embedder(y)
        c = t_emb + y_emb
        x = self.x_embedder(x) + self.pos_embed
        mask = x.new_ones((x.shape[0], 1, 1))
        if self.training and self.p_to_a_dropout > 0:
            mask.bernoulli_(1 - self.p_to_a_dropout)
        feature = None
        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                tokens = (
                    self.y_token_proj(y_emb)
                    .unsqueeze(1)
                    .repeat(1, self.in_context_len, 1)
                )
                x = torch.cat([tokens + self.in_context_posemb, x], dim=1).contiguous()
            has_context = self.in_context_len > 0 and i >= self.in_context_start
            rope = self.feat_rope_incontext if has_context else self.feat_rope
            x = block(x, c, rope, mask, p_to_a_scale)
            if return_persistent and i + 1 == self.feature_layer:
                image_tokens = x[:, self.in_context_len :] if has_context else x
                feature = image_tokens[..., self.persistent_start :]
        if self.in_context_len > 0:
            x = x[:, self.in_context_len :].contiguous()
        output = self.unpatchify(self.final_layer(x, c), self.patch_size)
        if return_persistent:
            if feature is None:
                raise ValueError(
                    "Persistent features require a contraction before the minimum width"
                )
            return output, feature
        return output


B_WIDTHS = (1024, 896, 832, 768, 640, 576, 512, 512, 576, 704, 896, 1024)
L_WIDTHS = (1600, 1472, 1408, 1280, 1152, 1088, 1088, 960, 896, 832, 832, 768, 704, 640, 640, 576, 576, 512, 640, 768, 896, 1088, 1280, 1600)
H_WIDTHS = (2000, 1920, 1840, 1760, 1600, 1520, 1440, 1360, 1360, 1280, 1200, 1120, 1040, 1040, 1040, 960, 880, 800, 800, 800, 720, 720, 720, 640, 800, 880, 1040, 1120, 1280, 1520, 1680, 2000)

MODEL_CONFIGS = {
    "PerF-B/16": dict(
        input_size=256,
        patch_size=16,
        hidden_size=1024,
        cond_dim=768,
        width_schedule=B_WIDTHS,
        p_to_a_rank=128,
        head_dim=64,
        bottleneck_dim=128,
        in_context_start=4,
    ),
    "PerF-L/16": dict(
        input_size=256,
        patch_size=16,
        hidden_size=1600,
        cond_dim=1024,
        width_schedule=L_WIDTHS,
        p_to_a_rank=128,
        head_dim=64,
        bottleneck_dim=128,
        in_context_start=8,
    ),
    "PerF-H/16": dict(
        input_size=256,
        patch_size=16,
        hidden_size=2000,
        cond_dim=1280,
        width_schedule=H_WIDTHS,
        p_to_a_rank=192,
        head_dim=80,
        bottleneck_dim=256,
        in_context_start=10,
    ),
    "PerF-H/32": dict(
        input_size=512,
        patch_size=32,
        hidden_size=2000,
        cond_dim=1280,
        width_schedule=H_WIDTHS,
        p_to_a_rank=192,
        head_dim=80,
        bottleneck_dim=256,
        in_context_start=10,
    ),
}


def build_model(name, **kwargs):
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {name!r}; choose from {tuple(MODEL_CONFIGS)}")
    config = dict(MODEL_CONFIGS[name])
    config.update(kwargs)
    return PerF(**config)
