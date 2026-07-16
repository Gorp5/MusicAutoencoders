'''
Modified from the myna repository https://github.com/ghost-signal/myna
'''

from torch.utils.checkpoint import checkpoint
from models.PositionalEmbeddings import *
from torch import nn
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import torch
import torch.nn as nn
import math

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Rotary2D(nn.Module):
    def __init__(self, dim_head, rope_on_x=True, rope_on_y=True, rope_base=8192.0):
        super().__init__()
        if not (rope_on_x or rope_on_y):
            raise ValueError("At least one axis must be enabled for RoPE.")
        if dim_head % 2 != 0:
            raise ValueError("dim_head must be even for rotary embeddings.")

        self.dim_head = dim_head
        self.rope_on_x = rope_on_x
        self.rope_on_y = rope_on_y
        self.rope_base = rope_base

        # Determine slices for each axis
        self.axis_slices = []
        if rope_on_x and rope_on_y:
            if dim_head % 4 != 0:
                raise ValueError("dim_head must be divisible by 4 when both axes are enabled.")
            half = dim_head // 2
            self.axis_slices = [("x", slice(0, half)), ("y", slice(half, dim_head))]
        elif rope_on_x:
            self.axis_slices = [("x", slice(0, dim_head))]
        elif rope_on_y:
            self.axis_slices = [("y", slice(0, dim_head))]

        # Precompute exp for each axis
        for axis_name, axis_slice in self.axis_slices:
            axis_dim = axis_slice.stop - axis_slice.start
            half = axis_dim // 2
            exp = torch.arange(0, half).float() / half
            self.register_buffer(f"rope_exp_{axis_name}", exp)

    def _axis_cos_sin(self, coords_axis, axis_name):
        exp = getattr(self, f"rope_exp_{axis_name}")
        log_base = math.log(self.rope_base)
        inv_freq = torch.exp(-exp * log_base)

        freqs = coords_axis.unsqueeze(-1) * inv_freq
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin

    def apply_rotary(self, t, cos, sin):
        # t: [..., dim], dim even
        t1, t2 = t[..., ::2], t[..., 1::2]
        return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)

    def _apply_axis(self, tensor, axis_slice, cos, sin, axis_dim):
        left = tensor[..., :axis_slice.start]
        mid = tensor[..., axis_slice]
        right = tensor[..., axis_slice.stop:]

        # Move the rotation axis to last dim for broadcasting
        mid_perm = mid.transpose(axis_dim, -2)
        rotated = self.apply_rotary(mid_perm, cos, sin)
        rotated = rotated.transpose(axis_dim, -2)
        return torch.cat([left, rotated, right], dim=-1)

    def forward(self, q, k, coords):
        if coords is None:
            return q, k

        B, H, N, D = q.shape

        coords = coords.view(B, N, 2)
        coords = coords.unsqueeze(1)

        q_out, k_out = q, k

        for axis_name, axis_slice in self.axis_slices:
            axis_index = 0 if axis_name == "x" else 1

            axis_coords = coords[..., axis_index]

            cos, sin = self._axis_cos_sin(axis_coords, axis_name)

            # cos/sin: [B, 1, N, half_dim]

            def apply(t):
                t_axis = t[..., axis_slice]

                t1 = t_axis[..., ::2]
                t2 = t_axis[..., 1::2]

                rotated = torch.cat([
                    t1 * cos - t2 * sin,
                    t1 * sin + t2 * cos
                ], dim=-1)

                return torch.cat([
                    t[..., :axis_slice.start],
                    rotated,
                    t[..., axis_slice.stop:]
                ], dim=-1)

            q_out = apply(q_out)
            k_out = apply(k_out)

        return q_out, k_out
    

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,
                 rope_base=-1, rope_on_x=False, rope_on_y=False):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.rotary = Rotary2D(
            dim_head=dim_head,
            rope_on_x=rope_on_x,
            rope_on_y=rope_on_y,
            rope_base=rope_base
        ) if rope_on_x or rope_on_y else None

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, alibi_bias=None, coords=None, mask=None):
        B, N, D = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if self.rotary is not None:
            q_cls, q_tokens = q[:, :, :1], q[:, :, 1:]
            k_cls, k_tokens = k[:, :, :1], k[:, :, 1:]

            q_tokens, k_tokens = self.rotary(q_tokens, k_tokens, coords)

            q = torch.cat([q_cls, q_tokens], dim=2)
            k = torch.cat([k_cls, k_tokens], dim=2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=alibi_bias,  # additive bias works here
            dropout_p=0.0,
            is_causal=False
        )

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 clamping=None,
                 alibi_on_x=False,
                 alibi_on_y=False,
                 learned_alibi_slopes=False,
                 use_rope_x=False,
                 use_rope_y=False,
                 rope_base=4096,
                 use_rope_double_frequency=False):

        super().__init__()

        if alibi_on_x or alibi_on_y:
            self.alibi_2d = Alibi2DBias(heads, alibi_on_x=alibi_on_x, alibi_on_y=alibi_on_y, clamping=clamping, learned_alibi_slopes=learned_alibi_slopes)
        else:
            self.alibi_2d = None

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, rope_base=rope_base, rope_on_x=use_rope_x, rope_on_y=use_rope_y),
                FeedForward(dim, mlp_dim)
            ]))

    def compute_alibi_with_cls(self, alibi_bias_2d, has_cls: bool):
        if not has_cls:
            return alibi_bias_2d

        B, H, N, _ = alibi_bias_2d.shape
        device = alibi_bias_2d.device
        dtype = alibi_bias_2d.dtype

        # Create full zero tensor and append biases into lower right slot
        full = torch.zeros((B, H, N + 1, N + 1), device=device, dtype=dtype)
        full[:, :, 1:, 1:] = alibi_bias_2d
        return full

    def _layer_forward(self, x, attn, ff, alibi_bias, coords, mask):
        x = attn(x, alibi_bias, coords, mask) + x
        x = ff(x) + x
        return x

    def forward(self, x, coords=None, cls=True, mask=None, checkpointing=False):
        if self.alibi_2d is not None:
            alibi_bias = self.alibi_2d(coords)
            alibi_bias = self.compute_alibi_with_cls(alibi_bias, has_cls=cls)
        else:
            alibi_bias = None

        for attn, ff in self.layers:
            if checkpointing:
                x = checkpoint(
                    self._layer_forward,
                    x,
                    attn,
                    ff,
                    alibi_bias,
                    coords,
                    mask,
                    use_reentrant=False
                )
            else:
                x = self._layer_forward(
                    x,
                    attn,
                    ff,
                    alibi_bias,
                    coords,
                    mask
                )

        return self.norm(x)

class StaticEmbeddings(nn.Module):
    def __init__(self, shape, image_height, image_width, patch_height, patch_width, d_model,
                 use_sinusoidal_x=False,
                 use_sinusoidal_y=False,
                 use_sinusoidal_raster=False,
                 use_learned_encoding_y=False,
                 use_learned_encoding_x=False,
                 device="cuda"):

        super().__init__()

        self.height_in_patches = image_height // patch_height
        self.width_in_patches = image_width // patch_width

        self.use_learned_encoding_y = use_learned_encoding_y
        self.use_learned_encoding_x = use_learned_encoding_x
        self.use_sinusoidal_x = use_sinusoidal_x
        self.use_sinusoidal_y = use_sinusoidal_y
        self.use_sinusoidal_raster = use_sinusoidal_raster

        if use_learned_encoding_y:
            self.y_pos_embedding = nn.Parameter(torch.zeros(1, self.height_in_patches, 1, d_model)).to(device=device)

        if use_learned_encoding_x:
            self.x_pos_embedding = nn.Parameter(torch.zeros(1, 1, self.width_in_patches, d_model)).to(device=device)

        if use_sinusoidal_x and use_sinusoidal_y:
            self.sinusoidal_embeddings = self.generate_sinusoidal_2d(self.height_in_patches, self.width_in_patches, d_model).unsqueeze(0).to(device=device)
        elif use_sinusoidal_x:
            x_positions = torch.arange(self.width_in_patches, device=device, dtype=torch.float32).unsqueeze(1)
            x_emb = self._generate_sinusoidal_1d(x_positions, d_model)
            self.sinusoidal_embeddings = x_emb.unsqueeze(0).expand(self.height_in_patches, -1, -1).unsqueeze(0).to(device=device)
        elif use_sinusoidal_y:
            y_positions = torch.arange(self.height_in_patches, device=device, dtype=torch.float32).unsqueeze(1)
            y_emb = self._generate_sinusoidal_1d(y_positions, d_model)
            self.sinusoidal_embeddings = y_emb.unsqueeze(1).expand(-1, self.width_in_patches, -1).unsqueeze(0).to(device=device)
        elif use_sinusoidal_raster:
            positions = torch.arange(self.height_in_patches * self.width_in_patches, device=device, dtype=torch.float32).unsqueeze(1)
            raster_emb = self._generate_sinusoidal_1d(positions, d_model).to(device=device)
            self.sinusoidal_embeddings = raster_emb.view(self.height_in_patches, self.width_in_patches, -1).unsqueeze(0).to(device=device)

        self.shape = shape

    def forward(self, x):
        if self.use_learned_encoding_y:
            x += self.y_pos_embedding

        if self.use_learned_encoding_x:
            x += self.x_pos_embedding

        if self.use_sinusoidal_x or self.use_sinusoidal_y or self.use_sinusoidal_raster:
            x += self.sinusoidal_embeddings

        return x

    def generate_sinusoidal_2d(self, height, width, embed_dim):
        assert embed_dim % 2 == 0, "Embedding dimension must be divisible by 2"
        embed_dim = embed_dim // 2

        # Generate x positions
        x_positions = torch.arange(width, dtype=torch.float32).unsqueeze(1)
        x_embedding = self._generate_sinusoidal_1d(x_positions, embed_dim)

        # Generate y positions
        y_positions = torch.arange(height, dtype=torch.float32).unsqueeze(1)
        y_embedding = self._generate_sinusoidal_1d(y_positions, embed_dim)

        # Expand and combine to H x W x E
        x_emb_expanded = x_embedding.unsqueeze(0).expand(height, -1, -1)
        y_emb_expanded = y_embedding.unsqueeze(1).expand(-1, width, -1)

        pos_embedding = torch.cat([x_emb_expanded, y_emb_expanded], dim=-1)
        return pos_embedding

    def _generate_sinusoidal_1d(self, positions, dim):
        device = positions.device
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(positions.size(0), dim, device=device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe  # N x E//2


class Myna(nn.Module):
    def __init__(self,
        image_size,
        channels = 1,
        patch_size = (16, 16),
        latent_space = 128,
        d_model = 384,
        depth = 12,
        heads = 6,
        mlp_dim = 1536,
        mask_ratio = 0.0,
        dim_head=64,
        use_rope_x = False,
        use_rope_y = False,
        use_alibi_x = False,
        use_alibi_y = False,
        use_learned_alibi_slopes = False,
        rope_base = 4096,
        latent_projection_method = "cls",
        use_sinusoidal_x = False,
        use_sinusoidal_y = False,
        use_sinusoidal_raster = True,
        use_learned_encoding_y = False,
        use_learned_encoding_x = False,
        use_rope_double_frequency = False,
        use_cls = True,
        clamping = None,
        device="cpu"):

        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.heads = heads
        self.num_patches_x = image_width // patch_width
        self.num_patches_y = image_height // patch_height
        self.num_patches = self.num_patches_x * self.num_patches_y

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        ).to(device)

        self.pos_embedding = StaticEmbeddings((1, self.num_patches, patch_dim), image_height, image_width, patch_height, patch_width, d_model,
                                                             use_sinusoidal_x=use_sinusoidal_x,
                                                             use_sinusoidal_y=use_sinusoidal_y,
                                                             use_sinusoidal_raster=use_sinusoidal_raster,
                                                             use_learned_encoding_y=use_learned_encoding_y,
                                                             use_learned_encoding_x=use_learned_encoding_x,
                                                             device=device).to(device)

        self.use_rope_x = use_rope_x
        self.use_rope_y = use_rope_y
        self.use_alibi_x = use_alibi_x
        self.use_alibi_y = use_alibi_y
        self.use_learned_alibi_slopes = use_learned_alibi_slopes
        self.rope_base = rope_base
        self.latent_projection_method = latent_projection_method
        self.use_sinusoidal_x = use_sinusoidal_x
        self.use_sinusoidal_y = use_sinusoidal_y
        self.use_sinusoidal_raster = use_sinusoidal_raster
        self.use_learned_encoding_y = use_learned_encoding_y
        self.use_learned_encoding_x = use_learned_encoding_x
        self.use_rope_double_frequency = use_rope_double_frequency

        self.needs_coordinates = use_rope_x or use_rope_y or use_alibi_x or use_alibi_y

        self.patch_coordinates = self.get_patch_coordinates(image_height, image_width, device)

        self.transformer = Transformer(d_model, depth, heads, dim_head, mlp_dim, clamping=clamping,
                                       use_rope_x=self.use_rope_x,
                                       use_rope_y=self.use_rope_y,
                                       rope_base=self.rope_base,
                                       learned_alibi_slopes = self.use_learned_alibi_slopes,
                                       use_rope_double_frequency=self.use_rope_double_frequency,
                                       alibi_on_x=use_alibi_x,
                                       alibi_on_y=use_alibi_y)

        self.to_latent = nn.Identity()

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model)) if use_cls else None

        self.mask_ratio = mask_ratio

        self.linear_head = nn.Linear(d_model, latent_space)

        self.to_cord = Rearrange("b (h w) f -> b h w f", w=self.num_patches_x, h=self.num_patches_y)
        self.from_cord = Rearrange("b h w f -> b (h w) f", w=self.num_patches_x, h=self.num_patches_y)

    def forward(self, img, mask=None, checkpointing=False):
        B, _, H, W = img.shape
        device = img.device

        x = self.to_patch_embedding(img)

        if mask is not None:
            mask = self.mask_to_patch_mask(mask).any(dim=-1)
            mask = mask.unsqueeze(1).expand(-1, self.heads, -1)
            mask = mask.reshape(B, -1)

        B, raster, F = x.shape

        x = self.to_cord(x)
        x = self.pos_embedding(x)
        x = self.from_cord(x)

        if self.mask_ratio > 0.0:
            unmasked = self.mask_inputs(x, mask, self.mask_ratio, device)
            x = x.gather(1, unmasked.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            if mask is not None:
                mask = mask.gather(1, unmasked)

            if self.needs_coordinates:
                coordinates = self.get_patch_coordinates(H, W, device).repeat(B, 1, 1)
                coordinates = coordinates.gather(1, unmasked.unsqueeze(-1).expand(-1, -1, 2))
            else:
                coordinates = None

        B, P, F = x.shape

        if self.cls_token is not None:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat([cls_tokens, x], dim=1)

            if mask is not None:
                cls_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
                mask = torch.cat((cls_mask, mask), dim=1)

        x = self.transformer(x, coords=coordinates, cls=self.cls_token is not None, mask=mask, checkpointing=checkpointing)

        if self.cls_token is not None:
            x = x[:, 0]
        else:
            if mask is not None:
                mask = mask.to(x.dtype)
                x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
            else:
                x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)

    def mask_inputs(self, x, mask, mask_ratio, device):
        B, N, _ = x.shape

        n_mask = int(mask_ratio * N)
        n_keep = max(1, N - n_mask)

        # Per-sample random permutations
        indices = torch.stack([
            torch.randperm(N, device=device)
            for _ in range(B)
        ])

        # Keep the last n_keep tokens for each sample
        unmasked_indices = indices[:, n_mask:n_mask + n_keep]

        return unmasked_indices.to(device=device)

    def mask_to_patch_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # (B, W)

        B, W = mask.shape
        ph, pw = self.patch_height, self.patch_width

        assert W % pw == 0, \
            "Mask spatial dimensions must be divisible by patch size"

        # Patchify exactly like the image
        patch_mask = rearrange(
            mask,
            "b (w pw) -> b (w) (pw)",
            pw=pw
        )

        return patch_mask

    def get_patch_coordinates(self, H, W, device=None):
        num_h = H // self.patch_height
        num_w = W // self.patch_width

        ys, xs = torch.meshgrid(
            torch.arange(num_h, device=device),
            torch.arange(num_w, device=device),
            indexing='ij'
        )

        coords = torch.stack([ys, xs], dim=-1).reshape(-1, 2)
        return coords

    def make_embedding_projection(self, patch_height, patch_width, patch_dim, dim):
        to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        return to_patch_embedding

