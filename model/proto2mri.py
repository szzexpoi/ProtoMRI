import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, unpack, pack

class Proto2MRI(nn.Module):
    """
    Simple mapping from prototype space to MRI space. This is a simple linear layer that takes in the prototype activations and outputs the predicted MRI voxel activations.
    """
    def __init__(self, num_prototype=1000, num_voxels=1000):
        super().__init__()
        # for compressing the spatial dimension
        self.attention = nn.Linear(num_prototype, 1)
        
        # linear mapping from prototype space to voxel space
        self.proto_vector = nn.Linear(num_prototype, num_voxels)

    def forward(self, x):
        # x: prototype similarity (B, K, num_prototype)
        B = x.shape[0]
        att = F.softmax(self.attention(x), dim=1) # (B, K, 1)
        proto_agg = (att * x).sum(dim=1) # (B, num_prototype)
        out = self.proto_vector(proto_agg) # (B, num_voxels)
        return out

    def get_att(self, x):
        att = F.softmax(self.attention(x), dim=1) # (B, K, 1)
        return att.squeeze(-1)        


class Proto2MRI_conv(nn.Module):
    """
    Simple mapping from prototype space to MRI space. 
    This is a version using convolutional layer for attention computation.
    """
    def __init__(self, num_prototype=1000, num_voxels=1000):
        super().__init__()
        # for compressing the spatial dimension
        self.attention = nn.Conv2d(num_prototype, 1, kernel_size=3, stride=1, padding=1)
        
        # linear mapping from prototype space to voxel space
        self.proto_vector = nn.Linear(num_prototype, num_voxels)

    def forward(self, x):
        # x: prototype similarity (B, K, num_prototype)
        B = x.shape[0]

        att_x = rearrange(x, 'b (h w) c -> b c h w', h=8, w=8)
        att = self.attention(att_x)
        att = rearrange(att, 'b c h w -> b (h w) c ')
        att = F.softmax(att, dim=1) # BxKx1

        proto_agg = (att * x).sum(dim=1) # (B, num_prototype)
        out = self.proto_vector(proto_agg) # (B, num_voxels)
        return out

    def get_att(self, x):

        att_x = rearrange(x, 'b (h w) c -> b c h w', h=8, w=8)
        att = self.attention(att_x)
        att = rearrange(att, 'b c h w -> b (h w) c ')
        att = F.softmax(att, dim=1) # BxKx1
        
        return att.squeeze(-1)        
