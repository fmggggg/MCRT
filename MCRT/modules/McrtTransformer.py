""" 
MCRT_Transformer
"""
from functools import partial
import torch
import torch.nn as nn
from torch.nn import AvgPool3d
from timm.models.layers import DropPath, trunc_normal_

#test
from MCRT.data_processor.dataset import Dataset
from MCRT.modules.cgcnn import GraphEmbeddings
from einops.layers.torch import Rearrange

def main():
    dataset_test=Dataset(r'D:\Projects\MyProjects\MCRT\MCRT\cifs\test','train',1023,4)
    batch=[dataset_test[i] for i in range(32)]
    collated_batch=dataset_test.collate(batch)
    # print(collated_batch["padded_distance_matrices"].shape)
    GraphEmbeddingsmodel=GraphEmbeddings(atom_fea_len=64, nbr_fea_len=41, max_graph_len=1023, hid_dim=512,is_finetune=False, n_conv=3,mask_probability=0.15)
    new_atom_fea, atom_label,atm_label, mask=GraphEmbeddingsmodel(atom_num=collated_batch["atom_num"], nbr_idx=collated_batch["nbr_idx"], nbr_fea=collated_batch["nbr_fea"], \
                                crystal_atom_idx=collated_batch["crystal_atom_idx"], atm_list=collated_batch["atm_list"])
    McrtTransformer=MCRT_Transformer(
        num_blocks=12,
        dim=512,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    )
    new_atom_fea=McrtTransformer(new_atom_fea,mask)
    print(new_atom_fea)

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

class MCRT_Transformer(nn.Module):
    def __init__(
        self,
        num_blocks,
        dim,
        num_heads,
        img_size, 
        patch_size, 
        in_chans=3, 
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

        self.patch_embed_1d = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=dim,
        )
        self.patch_embed_2d = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=dim,
        )
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

    def image_embed_1d(self, _x,):
        """

        :param _x: batch images, Tensor [B, C, H, W]
        :return:
            x:  Tensor [B, num_patches, hid_dim],
        """

        B, _, _, _,= _x.shape
        x = self.patch_embed_1d(_x)  # [B, num_patches, embed_dim]
        x_mask = torch.ones(x.shape[:2]).to(x)  # [B, num_patches]
        return x,x_mask
    
    def image_embed_2d(self, _x,):
        """

        :param _x: batch images, Tensor [B, C, H, W]
        :return:
            x:  Tensor [B, num_patches, hid_dim],
        """

        B, _, _, _,= _x.shape
        x = self.patch_embed_2d(_x)  # [B, num_patches, embed_dim]
        x_mask = torch.ones(x.shape[:2]).to(x)  # [B, num_patches]
        return x,x_mask
    
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class PatchEmbed2D(nn.Module):
    """ 2D Image to Patch Embedding """

    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=768):
        """
        Args:
            img_size (int or tuple): size of the image (height, width)
            patch_size (int or tuple): size of each patch (patch_height, patch_width)
            in_chans (int): number of input channels (e.g., 3 for RGB images)
            embed_dim (int): dimension of the embedding vectors
        """
        super().__init__()
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)  # (height, width)
        else:
            self.img_size = img_size

        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)  # (patch_height, patch_width)
        else:
            self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        patch_dim =self.patch_size[0] * self.patch_size[1] * self.in_chans
        self.proj = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size[0], p2=self.patch_size[1]),
            # nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.embed_dim),
            # nn.LayerNorm(self.embed_dim),
        )
        
        self.pos_embedding = posemb_sincos_2d(
            h = self.img_size[0] // self.patch_size[0],
            w = self.img_size[1] // self.patch_size[1],
            dim = embed_dim,
        ) 

    def forward(self, x):
        """
        Args:
            x (tensor): Tensor, shape [batch_size, channels, height, width]

        Returns:
            tensor: Tensor, shape [batch_size, num_patches, embed_dim]
        """
        x = self.proj(x)  # Apply the projection to flatten and embed
        x += self.pos_embedding.to(x.device, dtype=x.dtype)
        return x

if __name__ == "__main__":
    main()  