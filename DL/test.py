import matplotlib.pyplot as plt
from torchvision import datasets, transforms,models
from PIL import Image
import numpy as np
# a=[1,2,3]
# b=[4,5,6]
# plt.plot(a,b)
# plt.show()

# alexnet=models.alexnet(pretrained=False)
# print(alexnet)

# img=Image.open('21_training.tif').convert('RGB')
# img=np.array(img)
# print(img.shape)
# transforms=transforms.Compose(
#     [
#         transforms.ToPILImage(),
#         transforms.Resize((160,240)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,),(0.5,)) #归一化，均值和方差，三个通道（0.5，）表示三个通道的均值和方差都是0.5
#     ]
# )
# img=transforms(img)
# print(img.shape)
# print(img)

# import torch
#
# # 创建一个随机张量，形状为 (batch_size=2, channels=3, height=8, width=8)
# feature_map = torch.rand(2, 3, 8, 8)
#
# # 定义小块大小和步幅
# patch_size = 2
# stride = 2
# unfolded = feature_map.unfold(2, patch_size, stride)
# print("Shape of unfolded:", unfolded.shape)  # 输出形状为 (2, 3, 4, 4, 2, 2)
# unfolded = feature_map.unfold(3, patch_size, stride)
# print("Shape of unfolded:", unfolded.shape)  # 输出形状为 (2, 3, 4, 4, 2, 2)
# # 使用 unfold 获取小块
# unfolded = feature_map.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
# print("Shape of unfolded:", unfolded.shape)  # 输出形状为 (2, 3, 4, 4, 2, 2
# # 调整形状
# unfolded = unfolded.contiguous().view(2, 3, -1, patch_size, patch_size)
#
# # 展平每个补丁
# patch_dim = 3 * patch_size * patch_size
# patches = unfolded.view(2, -1, patch_dim)
#
# print("Shape of patches:", patches.shape)  # 输出形状为 (2, 16, 12)

# import torch
# a=torch.rand(2,3,3)
# print(a)
# b=torch.rand(3,3)
# print(b)
# c=a+b
# print(c) #广播机制，b被广播成(2,3,3)与a相加
# d=torch.rand(2,3,3,3)
# print(d)

# def a(a:int=1,b:int=3)->int:
# #     return (a+b)*1.0
# # c=a()
# # d=a(2,3)
# # print(c,d)

import torch
import math
import torch.nn.functional as F
# 创建一个张量
tensor = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])

# 创建一个布尔掩码
mask = torch.tensor([[False, True, False],
                     [True, False, True]])

# 使用 mask_fill 填充
filled_tensor = tensor.masked_fill(~mask, float('-inf'))

print(filled_tensor)

def to_2tuple(x):
    return tuple([x] * 2)
print(to_2tuple(3))

from PIL import Image

# 创建二值图像（模式 '1'）
binary_image = Image.new('1', (100, 100), color=1)  # 全部为白色
binary_image.save('binary_image.png')

# 创建灰度图像（模式 'L'）
gray_image = Image.new('L', (100, 100), color=128)  # 中间灰色
gray_image.save('gray_image.png')

# 打印模式
print(f"Binary Image Mode: {binary_image.mode}")  # 输出 '1'
print(f"Gray Image Mode: {gray_image.mode}")       # 输出 'L'


def uncertainty_map(x,num_classes):  #计算不确定性
    prob = F.softmax(x,dim=1)      #在dim=1上进行softmax,x.shape=[batch_size,num_classes,H,W]
    log_prob = torch.log2(prob + 1e-6)
    entropy = -1 * torch.sum(prob * log_prob, dim=1) / math.log2(num_classes)
    one = torch.ones_like(entropy, dtype=torch.int8)
    zero = torch.zeros_like(entropy, dtype=torch.int8)
    entropy = torch.where(entropy>=0.001,one,zero)
    return entropy

x=torch.rand(2,3,4,4)
num_classes=3
en=uncertainty_map(x,num_classes)
print(en.shape)  #torch.Size([2, 4, 4])

a=[1,2,3]
b=[4,5,6]
c=a+b
print(c)  #[1, 2, 3, 4, 5, 6]
#[5, 7, 9]
a=[1,2,3]
b=[4,5,6]
c=[i+j for i,j in zip(a,b)] #列表生成式
