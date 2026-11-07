import torch
from torch import nn

class EEGModel(nn.Module):
    """Fixed window T. Inputs [B,C,T], IDs [B,C], availability [B,C], recon [B,C,T].
    Returns [B,C,T]. Frozen representation [B,D] pools visible electrode memory.
    Queries contain electrode identity only; no target residual/copy path.
    """
    def __init__(self, time=64, dim=128, layers=2):
        super().__init__()
        self.temporal = nn.Sequential(nn.Conv1d(1,32,7,padding=3),nn.GELU(),nn.Conv1d(32,dim,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        self.embedding = nn.Embedding(22,dim)
        self.spatial = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,4,dim*2,dropout=0.,batch_first=True),layers,enable_nested_tensor=False)
        self.query = nn.MultiheadAttention(dim,4,dropout=0.,batch_first=True)
        self.head = nn.Linear(dim,time)

    def encode(self, signal, channel_ids, availability_mask, reconstruction_mask):
        visible = availability_mask & ~reconstruction_mask.any(-1)
        if not visible.any(-1).all():
            raise ValueError('At least one visible electrode required')
        safe = torch.where(visible[...,None],signal,0)
        b,c,t = safe.shape
        z = self.temporal(safe.reshape(b*c,1,t)).reshape(b,c,-1)
        z = self.spatial(z+self.embedding(channel_ids),src_key_padding_mask=~visible)
        pooled = torch.where(visible[...,None],z,0).sum(1)/visible.sum(1,keepdim=True)
        return z,visible,pooled

    def forward(self, signal, channel_ids, availability_mask, reconstruction_mask):
        z,visible,_ = self.encode(signal,channel_ids,availability_mask,reconstruction_mask)
        queries = self.embedding(channel_ids)
        decoded,_ = self.query(queries,z,z,key_padding_mask=~visible,need_weights=False)
        return self.head(decoded)

    def representation(self, signal, channel_ids, availability_mask):
        return self.encode(signal,channel_ids,availability_mask,torch.zeros_like(signal,dtype=torch.bool))[2]
