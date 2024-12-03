#猫狗分类
#数据集：https://www.kaggle.com/c/dogs-vs-cats/data
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import time

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
class cat_dog_dataset(Dataset):
    def __init__(self,dir):
        self.dir=dir
        self.files=os.listdir(dir)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        file=self.files[idx]
        fn=os.path.join(self.dir,file)
        img=Image.open(fn).convert('RGB')
        img=transform(img) #转换为张量,调整大小（3,224,224）
        img=img.reshape(-1,224,224)
        y=0 if 'cat' in file else 1
        return img,y

batch_size=20
train_dir='training_set'
train_dataset=cat_dog_dataset(train_dir)
test_dir='test_set'
test_dataset=cat_dog_dataset(test_dir)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

print(len(train_loader.dataset),len(test_loader.dataset))
class catdog(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,64,5,2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,128,3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*12*12,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,2),
        )
    def forward(self,x):
        out=self.features(x)
        out=out.reshape(x.size[0],-1) #x.size[0]是batch
        out=self.classifier(out)
        return out


model=catdog().to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9) #随机梯度下降,动量0.9,表示在更新参数时，不仅考虑当前梯度，还考虑之前的梯度，之前梯度的权重是0.9，当前梯度是0.1，减少震荡，加速收敛
model.train()
for ep in range(30):
    ep_loss=0
    for i,(x,y) in enumerate(train_loader):
        x,y=x.to(device),y.to(device)
        pre_y=model(x)
        loss=nn.CrossEntropyLoss()(pre_y,y.long()) #y.long()将y转换为长整型,是损失函数要求
        ep_loss+=loss*x.size(0) #loss是损失函数的平均值，乘以batch_size得到总的损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

correct=0
model.eval()
with torch.no_grad():
    for x,y in test_loader:
        x,y=x.to(device),y.to(device)
        pre_y=model(x)
        pre_y=pre_y.argmax(dim=1)
        pre_y=torch.argmax(pre_y,dim=1)
        correct+=(pre_y==y).sum().item()
        correct+=(pre_y==y).long().sum()






