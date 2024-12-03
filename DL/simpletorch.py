import torch
import torch.nn as nn
class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,20,5),
            nn.ReLU(),
            nn.Conv2d(20,10,5),
        )
        #self.pool=nn.MaxPool2d(2,2) __init__中定义相当于搭建网络结构，用nn.MaxPool2d(2,2)模块定义池化层比较好
        self.avgpool = nn.AdaptiveAvgPool2d((32,32))
        self.fc=nn.Linear(10*32*32,2)
    def forward(self,x):
        out=self.features(x)
        out=torch.max_pool2d(out,2,2) #forward中定义数据流向,直接调用torch的函数比较好
        out=self.avgpool(out)
        out=out.reshape(x.shape[0],-1)
        # out = self.dropout(out)
        out=self.fc(out)
        return out
model=mymodel()
print(model)
# mymodel(
#   (features): Sequential(
#     (0): Conv2d(3, 20, kernel_size=(5, 5), stride=(1, 1))
#     (1): ReLU()
#     (2): Conv2d(20, 10, kernel_size=(5, 5), stride=(1, 1))
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(32, 32))
#   (fc): Linear(in_features=10240, out_features=2, bias=True)
# )
x=torch.randn(32,3,64,64)
pre_y=model(x)
print(pre_y.shape)
# torch.Size([32, 2])

#访问模型参数
param_num=0
for param in model.parameters():
    print(param.shape)
    param_num+=param.numel()
print(param_num)
# 冻结模型某一层参数
# for param in model.features.parameters():
#     param.requires_grad=False

print(type(model.parameters())) #<class 'generator'>
print(type(model.named_parameters())) #<class 'generator'>
print(type(model.named_children())) #<class 'generator'>
print(type(model.named_modules())) #<class 'generator'>
print(type(model.named_buffers())) #<class 'generator'>
print(type(model.state_dict())) #<class 'dict'>

for k,param in model.state_dict().items():
    print(k,param.shape)

#保存和加载模型
#保存参数
torch.save(model.state_dict(),'model.pth')
#加载参数
mymodel = mymodel() #因为只保存了参数，所以要重新定义模型

param=torch.load('model.pth')
mymodel.load_state_dict(param)

#上面两行等于
#mymodel.load_state_dict(torch.load('model.pth'))

#或者直接保存模型
torch.save(model,'model.pth')
model = torch.load('model.pth')
