import torch
import torch.nn as nn
#调用nn.transformer模块
model=nn.Transformer(d_model=512,nhead=8,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=2048,dropout=0.1)
src=torch.rand(10,32,512)
tgt=torch.rand(20,32,512)
out=model(src,tgt)
print(out.shape)
print(model)
#打印参数量
param_num=0
for param in model.parameters():
    print(param.shape)
    param_num+=param.numel()
print(param_num)