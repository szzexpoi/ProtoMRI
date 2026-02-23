import torch
import torch.nn as nn
import torch.nn.functional as F

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
