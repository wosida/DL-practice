#AutoEncoder
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_size):
        super().__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        self.linear2=nn.Linear(hidden_size,latent_size)

    def forward(self,x):
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        return x

class Decoder(nn.Module):
    def __init__(self,latent_size,hidden_size,output_size):
        super().__init__()
        self.linear1=nn.Linear(latent_size,hidden_size)
        self.linear2=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x=F.relu(self.linear1(x))
        x=F.sigmoid(self.linear2(x))
        return x

class AE(nn.Module):
    def __init__(self,input_size,hidden_size,latent_size,output_size):
        super().__init__()
        self.encoder=Encoder(input_size,hidden_size,latent_size)
        self.decoder=Decoder(latent_size,hidden_size,output_size)

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

input_size=28*28
hidden_size=128
latent_size=16
output_size=28*28

model=AE(input_size,hidden_size,latent_size,output_size)
input=torch.randn(1,input_size)
out=model(input)
print(out.shape)

loss_MSE=torch.nn.MSELoss(reduction='sum')
dataset=datasets.MNIST(root='./MNIST',train=True)
dataloder=DataLoader(dataset,batch_size=64,shuffle=True)
epoches=100
optimizer=optim.Adam(model.parameters(),lr=0.001)
for i in range(epoches):
    for img,_ in dataloder:
        img=img.view(img.size(0),-1)
        output=model(img)
        loss=loss_MSE(output,img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch:{i},loss:{loss.item()}')


