import torch
import torch.nn as nn
from einops import rearrange

class Attn(nn.Module):
    def __init__(self,emb_size,seq_len,heads=8):
        super().__init__()
        self.qkv=nn.Linear(emb_size,3*emb_size)
        self.heads=heads
        self.seq_len=seq_len
        self.embedding_size=emb_size


    def forward(self,x): #x shape: [batch_size,seq_len,emb_size]
        qkv=self.qkv(x) #shape: [batch_size,seq_len,3*emb_size]
        q,k,v=torch.chunk(qkv,3,dim=-1)
        q,k,v=[rearrange(i,'b n (h d)-> b h n d',h=self.heads) for i in [q,k,v]]
#shape: [batch_size,heads,seq_len,emb_size//heads]
        dots=torch.einsum('bhid,bhjd->bhij',q,k)
        scaling_factor=self.embedding_size**0.5  #缩放因子
        attn=dots/scaling_factor
        attn=torch.softmax(attn,dim=-1) #shape: [batch_size,heads,seq_len,seq_len]
        out=torch.einsum('bhij,bhjd->bhid',attn,v)  #shape: [batch_size,heads,seq_len,emb_size//heads]
        out=rearrange(out,'b h n d->b n (h d)')
        return out

if __name__ == "__main__":
    a=Attn(512,128)
    x=torch.randn(1,128,512)
    print(a(x).shape)



