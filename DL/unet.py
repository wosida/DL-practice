#Unet用于图像分割，
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import random
device = torch.device('cuda' if torch.cuda.is_available() is True else 'cpu')
class OneModule(nn.Module): #定义一个卷积块
    def __init__(self,n1,n2):
        super().__init__()
        self.cnn= nn.Sequential(
            nn.Conv2d(n1,n2,3,padding=1,bias=False),
            nn.BatchNorm2d(n2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n2,n2,3,padding=1,bias=False),
            nn.BatchNorm2d(n2),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.cnn(x)

class Unet(nn.Module):
    def __init__(self,n1,n2):
        super().__init__()
        self.cnn1=OneModule(n1,64)
        self.cnn2=OneModule(64,128)
        self.cnn3=OneModule(128,256)
        self.cnn4=OneModule(256,512)
        self.bottleneck=OneModule(512,1024)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.ucnn4=OneModule(1024,512)
        self.ucnn3=OneModule(512,256)
        self.ucnn2=OneModule(256,128)
        self.ucnn1=OneModule(128,64)
        self.contr4 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.contr3 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.contr2 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.contr1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1_1 = nn.Conv2d(64,1,kernel_size=1) #1*1卷积，降维
    def forward(self,x): #torch.size([16,3,160,240])
        skip_cons=[]
        d1=x
        d1=self.cnn1(d1)
        skip_cons.append(d1)
        d1=self.pool(d1)
        d2=d1
        d2=self.cnn2(d2)
        skip_cons.append(d2)
        d2=self.pool(d2)
        d3=d2
        d3=self.cnn3(d3)
        skip_cons.append(d3)
        d3=self.pool(d3)
        d4=d3
        d4=self.cnn4(d4)
        skip_cons.append(d4)
        d4=self.pool(d4)
        bo=d4
        bo=self.bottleneck(bo)
        u4=bo
        u4=self.contr4(u4)
        u4=torch.cat((skip_cons[3],u4),dim=1)
        u4=self.ucnn4(u4)
        u3=u4
        u3=self.contr3(u3)
        u3=torch.cat((skip_cons[2],u3),dim=1)
        u3=self.ucnn3(u3)
        u2=u3
        u2=self.contr2(u2)
        u2=torch.cat((skip_cons[1],u2),dim=1)
        u2=self.ucnn2(u2)
        u1=u2
        u1=self.contr1(u1)
        u1=torch.cat((skip_cons[0],u1),dim=1)
        u1=self.ucnn1(u1)
        o= self.conv1_1(u1)
        return o

model=Unet(3,1).to(device)
# 创建一个虚拟输入张量
dummy_input = torch.randn(1, 3, 240, 240)  # batch_size=1, channels=3, height=256, width=256 shuru

# 运行模型
with torch.no_grad():  # 不需要计算梯度
    output = model(dummy_input)

# 打印输出形状
print("Output shape:", output.shape)
print(model.__class__.__name__)
print(model)
print(model.state_dict().keys()) #打印模型参数的键
print(model.cnn1)
print(model.cnn1.__class__.__name__)
print(model.cnn1.cnn)
#print(model[0]) #error,必须是Sequential的子类才有索引


param_num=0
for param in model.parameters():
    print(param.shape)
    param_num+=param.numel()
print(param_num)