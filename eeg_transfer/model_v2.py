"""Time-preserving V2; V1 remains byte-for-byte in model.py."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class EEGModelV2(nn.Module):
    """[B,C,T] -> [B,C,T]. Patches [B,C,N,D], N=ceil(T/patch_size).

    Whole-electrode masking: any target point makes its entire electrode invisible.
    At each temporal patch, spatial attention uses only visible electrodes. Query
    contains identity/time, with no raw-input residual. A temporal conv on visible
    patch features retains local context; D->patch_size restores ordered samples.
    """
    def __init__(self,time=250,dim=64,layers=2,patch_size=5):
        super().__init__()
        if time<1 or patch_size<1 or dim%4:
            raise ValueError('Invalid V2 dimensions')
        self.time=time; self.patch_size=patch_size
        self.temporal=nn.Sequential(nn.Conv1d(1,dim,patch_size,stride=patch_size),nn.GELU(),nn.Conv1d(dim,dim,3,padding=1),nn.GELU())
        self.embedding=nn.Embedding(22,dim)
        self.spatial=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,4,dim*2,dropout=0.,batch_first=True),layers,enable_nested_tensor=False)
        self.query=nn.MultiheadAttention(dim,4,dropout=0.,batch_first=True)
        self.head=nn.Linear(dim,patch_size)
        n=math.ceil(time/patch_size)
        position=torch.arange(n,dtype=torch.float32)[:,None]
        frequencies=torch.exp(torch.arange(0,dim,2)*(-math.log(10000.)/dim))
        pos=torch.zeros(n,dim); pos[:,0::2]=torch.sin(position*frequencies); pos[:,1::2]=torch.cos(position*frequencies)
        self.register_buffer('position',pos)

    def encode(self,signal,channel_ids,availability_mask,reconstruction_mask):
        b,c,t=signal.shape
        if t!=self.time:
            raise ValueError('V2 requires its configured window length')
        visible=availability_mask & ~reconstruction_mask.any(-1)
        if not visible.any(-1).all():
            raise ValueError('At least one visible electrode required')
        safe=torch.where(visible[...,None],signal,0)
        safe=F.pad(safe,(0,(-t)%self.patch_size))
        z=self.temporal(safe.reshape(b*c,1,-1)).transpose(1,2)
        n=z.shape[1]
        z=z.reshape(b,c,n,-1).permute(0,2,1,3)
        identities=self.embedding(channel_ids)[:,None]
        time=self.position[None,:,None,:]
        z=(z+identities+time).reshape(b*n,c,-1)
        keys=visible[:,None,:].expand(b,n,c).reshape(b*n,c)
        z=self.spatial(z,src_key_padding_mask=~keys)
        queries=(identities+time).expand(b,n,c,-1).reshape(b*n,c,-1)
        return z,keys,queries,n

    def forward(self,signal,channel_ids,availability_mask,reconstruction_mask):
        b,c,t=signal.shape
        z,keys,queries,n=self.encode(signal,channel_ids,availability_mask,reconstruction_mask)
        decoded,_=self.query(queries,z,z,key_padding_mask=~keys,need_weights=False)
        patches=self.head(decoded).reshape(b,n,c,self.patch_size)
        return patches.permute(0,2,1,3).reshape(b,c,-1)[...,:t]
