#VIT实现
from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce #这样才能把Rearrange和Reduce当作函数使用，写在nn.Sequential中
from einops import rearrange, repeat
from torch import Tensor
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle


class Patchembed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2 #224/16=14,14*14=196
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))#分类token,1*1*768,可学习,会被广播到所有位置
        self.positions = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
    def forward(self, x):
        b=x.shape[0]
        x=self.proj(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positions
        return x

print(Patchembed()(torch.randn(1,3,224,224)).shape) #torch.Size([1, 197, 768])

class Multihead_Attention_origin(nn.Module):
    def __init__(self,emb_size:int = 768,num_heads:int = 8,dropout:float=0):  #emb_size=768,num_heads=8,所以d_k=768/8=96,这里的emb_size是输入的维度
        super().__init__()
        self.emb_size=emb_size
        self.num_heads=num_heads 
        self.queries=nn.Linear(emb_size,emb_size)
        self.keys=nn.Linear(emb_size,emb_size)
        self.values=nn.Linear(emb_size,emb_size)
        #改进的代码，将qkv放在一个矩阵中，然后分割
       # self.qkv=nn.Linear(emb_size,emb_size*3)
        self.att_drop=nn.Dropout(dropout)
        self.projection=nn.Linear(emb_size,emb_size)
    def forward(self,x:Tensor,mask:Tensor=None)->Tensor:
        # split q,k,v in num_heads
        queries=rearrange(self.queries(x),"b n (h d)->b h n d",h=self.num_heads)
        keys=rearrange(self.keys(x),"b n (h d)->b h n d",h=self.num_heads)
        values=rearrange(self.values(x),"b n (h d)->b h n d",h=self.num_heads)#[b,heads,len,embed_size]

        #计算qkv
        qkv=self.qkv(x)
        #另一种分配方法
        #q,k,v=torch.chunk(qkv,3,dim=-1) #将qkv分割成三个张量,最后一个维度上分割，chunk是均分
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)  # 划分到num_heads个头上
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd,bhkd->bhqk',queries,keys)  #[b,heads,q_len,k_len]
        if mask is not None:
            fill_value=torch.finfo(torch.float32).min #设置 fill_value 为 -infinity（实际上是 float32 类型能表示的最小值），用于将某些分数置为非常低的值，从而在后续的 softmax 操作中，这些分数几乎会变成零。
            energy=energy.masked_fill(~mask,fill_value)#~mask: 这是对 mask 的按位取反操作。 energy 张量中对应于 ~mask 为 True 的位置替换为 fill_value（即负无穷大）。这样一来，对于那些被掩码所遮蔽的位置，其注意力权重会被设置为极小值，以确保在后续的 softmax 中这些位置的概率接近零
        scaling=(self.emb_size//self.num_heads)**(1/2)  #缩放因子,应该是emb_size//num_heads的开方
        att=F.softmax(energy,dim=-1)/scaling
        att=self.att_drop(att)
        #sum up over third axis
        out=torch.einsum('bhal,bhlv->bhav',att,values)
        out=rearrange(out,'b h n d->b n (h d)')
        out=self.projection(out)
        return out

class Multihead_Attention(nn.Module): #input_dim即emb_size
    def __init__(self,input_dim:int,num_heads:int,head_dim:int,output_dim:int,dropout:float=0.):
        super().__init__()
        assert input_dim%num_heads==0,f"input_dim {input_dim} should be divisible by num_heads {num_heads}"

        self.input_dim=input_dim
        self.num_heads=num_heads
        self.head_dim=head_dim
        self.inner_dim=head_dim*num_heads
        self.scale=self.head_dim**-0.5
        self.qkv=nn.Linear(self.input_dim,self.inner_dim*3,bias=False) #qkv形状为(input_dim,inner_dim*3)
        self.output_dim=output_dim #输出维度,一般等于输入维度或inner_dim

        self.att_drop=nn.Dropout(dropout)
        self.projection=nn.Linear(self.inner_dim,self.output_dim)
    def forward(self,x,mask=None):
        qkv=rearrange(self.qkv(x),'b n (qkv h d)->(qkv) b h n d',h=self.num_heads,qkv=3) #x的形状为(b,n,d),qkv的形状为(b,n,inner_dim*3),将qkv分成三个张量
        q,k,v=qkv[0],qkv[1],qkv[2]     #多头注意力机制中，x的维度从来没有分割过，只有qkv分割过
        energy=torch.einsum('bhid,bhjd->bhij',q,k)
        if mask is not None:
            fill_value=torch.finfo(torch.float32).min
            energy=energy.masked_fill(~mask,fill_value)
        att=F.softmax(energy,dim=-1)/self.scale
        att=self.att_drop(att)
        out=torch.einsum('bhij,bhjd->bhid',att,v)
        out=rearrange(out,'b h n d->b n (h d)')
        out=self.projection(out)
        return out


x=torch.randn(1,3,224,224)
patch_embed=Patchembed()
x=patch_embed(x)
print(x.shape) #torch.Size([1, 197, 768])
multihead_attention=Multihead_Attention(768,8,96,512)
x=multihead_attention(x)
print(x.shape) #torch.Size([1, 197, 512])






class ResidualAdd(nn.Module): #残差连接的实现方法：往残差块里传需要的函数，然后实现输入与函数输出相加，最后输出
    def __init__(self,fn):
        super().__init__()
        self.fn=fn

    def forward(self,x,**kwargs):
        res=x
        x=self.fn(x,**kwargs)
        x+=res
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim: int, expansion: int = 4, drop_p: float = 0.): #expansion是扩展维度倍数
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, expansion * input_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * input_dim, input_dim),
        )
    def forward(self, x):
        return self.net(x)

class TransformerEncoderBlock(nn.Sequential):  #nn.Sequential是一个容器，可以将模块按顺序排列,不需要forward函数
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(   #将Sequential的功能传给fn
                nn.LayerNorm(emb_size),
                Multihead_Attention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

a=torch.randn(1,3,224,224)
patch_embed=Patchembed()
print(sum([p.numel() for p in patch_embed.parameters()])) #参数量
a=patch_embed(a)
transformer_encoder_block=TransformerEncoderBlock(num_heads=8,head_dim=96,output_dim=768)
a=transformer_encoder_block(a)
print(sum([p.numel() for p in transformer_encoder_block.parameters()])) #参数量







