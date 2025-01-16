#深度监督举例,引入辅助loss
import torch.nn as nn
import torch
# 定义卷积块
# 包含3x3卷积+BN+relu
def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.Conv2d(out_planes, out_planes, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

#深度监督
class DSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=conv3x3_bn_relu(3,64)
        self.pool1=nn.MaxPool2d(2)
        self.conv2=conv3x3_bn_relu(64,128)
        self.pool2=nn.MaxPool2d(2)
        self.aufc1=nn.Linear(128,10)
        self.conv3=conv3x3_bn_relu(128,256)
        self.pool3=nn.MaxPool2d(2)
        self.conv4=conv3x3_bn_relu(256,512)
        self.pool4=nn.AdaptiveAvgPool2d((1,1))
        self.fc1=nn.Linear(512,1024)
        self.fc2=nn.Linear(1024,10)
    def forward(self,x):
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.pool2(x)
        aux1=nn.AdaptiveAvgPool2d((1,1))(x)
        aux1=aux1.view(aux1.size(0),-1)
        aux1=self.aufc1(aux1)
        x=self.conv3(x)
        x=self.pool3(x)
        x=self.conv4(x)
        x=self.pool4(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        y=self.fc2(x)
        return y,aux1

def train(epochs,trainloader):
    net=DSNet()
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(net.parameters())
    for epoch in range(epochs):
        for i,data in enumerate(trainloader,0):
            inputs,labels=data
            optimizer.zero_grad()
            outputs,aux1=net(inputs)
            loss1=criterion(outputs,labels)
            loss2=criterion(aux1,labels)
            loss=loss1+0.4*loss2
            loss.backward()
            optimizer.step()
            print(f'epoch:{epoch},step:{i},loss:{loss.item()}')





