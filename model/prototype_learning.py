import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
import numpy as np
from einops import rearrange, repeat, unpack, pack
from torch import einsum
from einops.layers.torch import Rearrange
import sys
sys.path.append('/home/shichen/project/dinov3/')
from dinov3.layers import SelfAttentionBlock, CausalSelfAttentionBlock, Mlp, LayerScale
from functools import partial

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def pack_one(t, pattern):
    return pack([t], pattern)

def identity(t, *args, **kwargs):
    return t

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
    
class DinoPro(nn.Module):
    """ Prototype learning model for DINO v3
    """
    def __init__(self, decoder_dim=768, encoder_depth=4, decoder_depth=4, num_head=8, num_prototypes=1000,
                 topk=32):
        super().__init__()
        self.encoder = nn.ModuleList([CausalSelfAttentionBlock(
                                                dim=decoder_dim,
                                                num_heads=num_head,
                                                ffn_ratio=4.0,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                act_layer=nn.GELU,
                                                ls_init_value=None,
                                                is_causal=True,
                                            )
                                            for _ in range(encoder_depth)])     
        
        self.decoder = nn.ModuleList([CausalSelfAttentionBlock(
                                                dim=decoder_dim,
                                                num_heads=num_head,
                                                ffn_ratio=4.0,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                act_layer=nn.GELU,
                                                ls_init_value=None,
                                                is_causal=True,
                                            )
                                            for _ in range(decoder_depth)])     

        # simple fixed latents for positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, decoder_dim), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed_from_grid(decoder_dim, np.arange(256)) 
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # learnable prototypes 
        self.prototypes = nn.Linear(decoder_dim, num_prototypes, bias=False)
        self.topk = topk
        

        # visual decoder for next-patch predictions (rasterized) 
        self.visual_readout = nn.Sequential(nn.Linear(decoder_dim, decoder_dim),
                                                nn.LayerNorm(decoder_dim)
                                                )

    def forward(self, img_feat):
        b, n, c = img_feat.shape
        x = img_feat + self.pos_embedding
        for block in self.encoder:
            x = block(x)

        # compute prototype similarity and top-k nearest prototypes
        proto_sim = self.prototypes(x) # (b, n, num_prototypes)
        
        # mask out prototypes beyond top-k
        topk_sim, topk_idx = torch.topk(proto_sim, self.topk, dim=-1) # (b, n, topk)
        mask = torch.full_like(proto_sim, float('-inf')) # (b, n, num_prototypes)
        mask.scatter_(-1, topk_idx, topk_sim) # (b, n, num_prototypes), only top-k are kept, others are -inf
        proto_sim = mask.softmax(dim=-1).to(x.device) # (b, n, num_prototypes), softmax over prototypes
        
        # transform the original features into a weighted sum of prototypes
        x_p = einsum('b n p, p c -> b n c', proto_sim, self.prototypes.weight) # (b, n, decoder_dim)

        # use the transformed features for next-patch prediction (maybe stick to simple/shallow decoder to encourage discriminative prototypes)
        for block in self.decoder:
            x_p = block(x_p)
        visual_pred = self.visual_readout(x_p)

        return visual_pred
    
    def get_prototype_similarity(self, img_feat):
        b, n, c = img_feat.shape
        x = img_feat + self.pos_embedding
        for block in self.encoder:
            x = block(x)

        proto_sim = self.prototypes(x) # (b, n, num_prototypes)
        # mask out prototypes beyond top-k
        topk_sim, topk_idx = torch.topk(proto_sim, self.topk, dim=-1) # (b, n, topk)
        mask = torch.full_like(proto_sim, float('-inf')) # (b, n, num_prototypes)
        mask.scatter_(-1, topk_idx, topk_sim) # (b, n, num_prototypes), only top-k are kept, others are -inf
        proto_sim = mask.softmax(dim=-1).to(x.device) # (b, n, num_prototypes), softmax over prototypes
        
        return proto_sim

    def get_latents(self, img_feat):
        b, n, c = img_feat.shape
        x = img_feat + self.pos_embedding
        for block in self.encoder:
            x = block(x)

        # compute prototype similarity and top-k nearest prototypes
        proto_sim = self.prototypes(x) # (b, n, num_prototypes)
        
        # mask out prototypes beyond top-k
        topk_sim, topk_idx = torch.topk(proto_sim, self.topk, dim=-1) # (b, n, topk)
        mask = torch.full_like(proto_sim, float('-inf')) # (b, n, num_prototypes)
        mask.scatter_(-1, topk_idx, topk_sim) # (b, n, num_prototypes), only top-k are kept, others are -inf
        proto_sim = mask.softmax(dim=-1).to(x.device) # (b, n, num_prototypes), softmax over prototypes
        
        # transform the original features into a weighted sum of prototypes
        x_p = einsum('b n p, p c -> b n c', proto_sim, self.prototypes.weight) # (b, n, decoder_dim)

        # use the transformed features for next-patch prediction (maybe stick to simple/shallow decoder to encourage discriminative prototypes)
        for block in self.decoder:
            x_p = block(x_p)
        visual_pred = self.visual_readout(x_p)

        return visual_pred

class SpatialDownsample2x(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.conv = nn.Conv2d(dim, dim_out, kernel_size, stride = 2, padding = kernel_size // 2)

    def forward(self, x):
        out = self.conv(x)
        return out

class spatial_downsample_block(nn.Module):
    def __init__(self, input, output):
        super(spatial_downsample_block, self).__init__()
        self.block = nn.Sequential(
                                SpatialDownsample2x(input, input),
                                nn.Conv2d(input, output,
                                        kernel_size= 1, stride= 1, padding=0),
                                nn.BatchNorm2d(output),
                                nn.GELU(), 
                                )
    
    def forward(self, x):
        return self.block(x)

class DinoPro_slim(nn.Module):
    """ Prototype learning model for DINO v3
    """
    def __init__(self, decoder_dim=768, encoder_depth=4, decoder_depth=4, num_head=8, num_prototypes=1000,
                 topk=32):
        super().__init__()
        self.visual_head = spatial_downsample_block(decoder_dim, decoder_dim)
        self.encoder = nn.ModuleList([CausalSelfAttentionBlock(
                                                dim=decoder_dim,
                                                num_heads=num_head,
                                                ffn_ratio=4.0,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                act_layer=nn.GELU,
                                                ls_init_value=None,
                                                is_causal=True,
                                            )
                                            for _ in range(encoder_depth)])     
        self.decoder_depth = decoder_depth
        if decoder_depth > 0:
            self.decoder = nn.ModuleList([CausalSelfAttentionBlock(
                                                    dim=decoder_dim,
                                                    num_heads=num_head,
                                                    ffn_ratio=4.0,
                                                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                    act_layer=nn.GELU,
                                                    ls_init_value=None,
                                                    is_causal=True,
                                                )
                                                for _ in range(decoder_depth)])     

        # simple fixed latents for positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, decoder_dim), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed_from_grid(decoder_dim, np.arange(64)) 
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # learnable prototypes 
        self.prototypes = nn.Linear(decoder_dim, num_prototypes, bias=False)
        self.topk = topk
        

        # visual decoder for next-patch predictions (rasterized) 
        self.visual_readout = nn.Sequential(nn.Linear(decoder_dim, decoder_dim*4),
                                                nn.LayerNorm(decoder_dim*4)
                                                )

    def forward(self, img_feat):
        b, n, c = img_feat.shape
        x = rearrange(img_feat, 'b (h w) c -> b c h w', h=16, w=16)
        x = self.visual_head(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=8, w=8)
        x = x + self.pos_embedding
        for block in self.encoder:
            x = block(x)

        # compute prototype similarity and top-k nearest prototypes
        proto_sim = self.prototypes(x) # (b, n, num_prototypes)
        
        # mask out prototypes beyond top-k
        topk_sim, topk_idx = torch.topk(proto_sim, self.topk, dim=-1) # (b, n, topk)
        mask = torch.full_like(proto_sim, float('-inf')) # (b, n, num_prototypes)
        mask.scatter_(-1, topk_idx, topk_sim) # (b, n, num_prototypes), only top-k are kept, others are -inf
        proto_sim = mask.softmax(dim=-1).to(x.device) # (b, n, num_prototypes), softmax over prototypes
        
        # transform the original features into a weighted sum of prototypes
        x_p = einsum('b n p, p c -> b n c', proto_sim, self.prototypes.weight) # (b, n, decoder_dim)

        # use the transformed features for next-patch prediction (maybe stick to simple/shallow decoder to encourage discriminative prototypes)
        if self.decoder_depth > 0:
            for block in self.decoder:
                x_p = block(x_p)
        visual_pred = self.visual_readout(x_p)

        return visual_pred
    
    def get_prototype_similarity(self, img_feat):
        b, n, c = img_feat.shape
        x = rearrange(img_feat, 'b (h w) c -> b c h w', h=16, w=16)
        x = self.visual_head(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=8, w=8)
        x = x + self.pos_embedding
        for block in self.encoder:
            x = block(x)

        proto_sim = self.prototypes(x) # (b, n, num_prototypes)
        # mask out prototypes beyond top-k
        topk_sim, topk_idx = torch.topk(proto_sim, self.topk, dim=-1) # (b, n, topk)
        mask = torch.full_like(proto_sim, float('-inf')) # (b, n, num_prototypes)
        mask.scatter_(-1, topk_idx, topk_sim) # (b, n, num_prototypes), only top-k are kept, others are -inf
        proto_sim = mask.softmax(dim=-1).to(x.device) # (b, n, num_prototypes), softmax over prototypes
        
        return proto_sim

    def get_latents(self, img_feat):
        b, n, c = img_feat.shape
        x = rearrange(img_feat, 'b (h w) c -> b c h w', h=16, w=16)
        x = self.visual_head(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=8, w=8)
        x = x + self.pos_embedding
        for block in self.encoder:
            x = block(x)

        # compute prototype similarity and top-k nearest prototypes
        proto_sim = self.prototypes(x) # (b, n, num_prototypes)
        
        # mask out prototypes beyond top-k
        topk_sim, topk_idx = torch.topk(proto_sim, self.topk, dim=-1) # (b, n, topk)
        mask = torch.full_like(proto_sim, float('-inf')) # (b, n, num_prototypes)
        mask.scatter_(-1, topk_idx, topk_sim) # (b, n, num_prototypes), only top-k are kept, others are -inf
        proto_sim = mask.softmax(dim=-1).to(x.device) # (b, n, num_prototypes), softmax over prototypes
        
        # transform the original features into a weighted sum of prototypes
        x_p = einsum('b n p, p c -> b n c', proto_sim, self.prototypes.weight) # (b, n, decoder_dim)

        # # # use the transformed features for next-patch prediction (maybe stick to simple/shallow decoder to encourage discriminative prototypes)
        # for block in self.decoder:
        #     x_p = block(x_p)

        # visual_pred = self.visual_readout(x_p)

        return x_p