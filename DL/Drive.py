#drive数据集增强及dataloader
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import random
import cv2

#数据集增强
#import numpy as np
import matplotlib.pyplot as plt

# 创建一个二值图像（0 和 1）
binary_image_01 = np.array([[0, 1], [1, 0]])

# 创建一个二值图像（0 和 255）
binary_image_255 = binary_image_01 * 255

# 显示图像
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Binary Image (0 and 1)')
plt.imshow(binary_image_01, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Binary Image (0 and 255)')
plt.imshow(binary_image_255, cmap='gray')
plt.axis('off')

plt.show()
