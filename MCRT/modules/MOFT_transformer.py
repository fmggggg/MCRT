""" 
MOFT_Transformer
"""
from functools import partial
import torch
import torch.nn as nn
from torch.nn import AvgPool3d
from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        assert C % self.num_heads == 0
        qkv = (
            self.qkv(x)  # [B, N, 3*C]
            .reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            )  # [B, N, 3, num_heads, C//num_heads]
            .permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, C//num_heads]
        )
        q, k, v = (
            qkv[0],  # [B, num_heads, N, C//num_heads]
            qkv[1],  # [B, num_heads, N, C//num_heads]
            qkv[2],  # [B, num_heads, N, C//num_heads]
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)  # [B, num_heads, N, N]
        attn = self.attn_drop(attn)

        x = (
            (attn @ v).transpose(1, 2).reshape(B, N, C)
        )  # [B, num_heads, N, C//num_heads] -> [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

class MOFT_Transformer(nn.Module):
    def __init__(
        self,
        num_blocks,
        dim,
        num_heads,
        grid_img_size,
        grid_patch_size,
        grid_in_chans,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.grid_img_size = grid_img_size
        self.grid_patch_size = grid_patch_size
        self.grid_in_chans = grid_in_chans
        self.dim = dim
        self.patch_embed_3d = PatchEmbed3D(#for grid
            img_size=self.grid_img_size,
            patch_size=self.grid_patch_size,
            in_chans=self.grid_in_chans,
            embed_dim=self.dim,
        )

        self.num_patches = self.patch_embed_3d.num_patches  
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.dim))

        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path_rate * i / (num_blocks - 1),
                act_layer=act_layer,
                norm_layer=norm_layer
            )
            for i in range(num_blocks)
        ])
        self.norm = norm_layer(dim)
        self.apply(self._init_weights)

    def forward(self, x, mask=None):
        for block in self.blocks:
            x, _ = block(x, mask=mask)
        x = self.norm(x)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def visual_embed_3d(self, _x,):
        """

        :param _x: batch images, Tensor [B, C, H, W, D]
        :return:
            x:  Tensor [B, max_image_len+1, hid_dim],
            x_mask: Tensor [B, max_image_len+1]],
        """

        B, _, _, _, _ = _x.shape
        x = self.patch_embed_3d(_x)  # [B, ph*pw*pd, embed_dim]
        # x = x.flatten(2).transpose(1, 2)

        # cls tokens
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, ph*pw*pd, embed_dim]

        # positional embedding
        x += self.pos_embed

        x_mask = torch.ones(x.shape[:2]).to(x)  # [B, ph*pw*pd]

        return x, x_mask

class PatchEmbed3D(nn.Module):
    """Image to Patch Embedding for 3D"""

    def __init__(
        self,
        img_size,  # minimum of H or W ex. 384
        patch_size,  # p -> length of fixed patch ex. 32
        in_chans=1,
        embed_dim=768,
        no_patch_embed_bias=False,
    ):
        super().__init__()

        assert img_size % patch_size == 0
        num_patches = (img_size**3) // (patch_size**3)
        self.img_size = img_size  # default: 30
        self.patch_size = patch_size  # default: 5
        self.num_patches = num_patches

        self.proj = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)",
                p1=patch_size,
                p2=patch_size,
                p3=patch_size,
            ),
            nn.Linear(patch_size * patch_size * patch_size * in_chans, embed_dim),
        )

    def forward(self, x):
        x = self.proj(x)  # [B, num_patches,
        return x  # [B, emb_dim, px, ph, pd]