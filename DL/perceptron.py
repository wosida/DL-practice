#感知机
import torch.nn as nn
import torch
X1=[2.49,0.50,2.73,3.47,1.38,1.03,0.59,2.25,0.15,2.73]
X2=[2.86,0.21,2.91,2.34,0.37,0.27,1.73,3.75,1.45,3.42]
Y=[1,0,1,1,0,0,0,1,0,1]
X1=torch.tensor(X1)
X2=torch.tensor(X2)
Y=torch.tensor(Y)
X=torch.stack([X1,X2],dim=1)#dim=1表示在第一维度上进行堆叠,形状是[10,2]

class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1=nn.Parameter(torch.Tensor([0.0]))
        self.w2=nn.Parameter(torch.Tensor([0.0]))
        self.b=nn.Parameter(torch.Tensor([0.0]))
    def f(self,x):
        t=self.w1*x[0]+self.w2*x[1]+self.b
        out=1.0/(1+torch.exp(t))
        return out
    def forward(self,x):
        pre_y=self.f(x)
        return pre_y

perceptron=Perceptron()
print(perceptron)
#设计优化器
optimizer=torch.optim.Adam(perceptron.parameters(),lr=0.1)
for ep in range(100):
    for (x,y) in zip(X,Y):
        print(x,y)
        pre_y=perceptron(x)
        y=torch.Tensor([y]) #将y转换为张量,因为y是一个数,先变成列表，即由torch.size[]变成torch.size[1]
        loss=nn.BCELoss()(pre_y,y)
        optimizer.zero_grad()#梯度清零,因为每次迭代都会累加梯度
        loss.backward()#反向传播并计算梯度
        optimizer.step()#更新参数

#f字符串打印感知机模型
s=f'学习的感知机为：y={perceptron.w1.item()}*x1+{perceptron.w2.item()}*x2+{perceptron.b.item()}'#item()方法将张量转换为Python数值
print(s)

#利用nn.Linear模块实现感知机
class Perceptron2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(2,1,bias=True)
    def forward(self,x):
        pre_y=self.fc(x)
        pre_y=torch.sigmoid(pre_y)
        return pre_y
perceptron2=Perceptron2()
optimizer2=torch.optim.Adam(perceptron2.parameters(),lr=0.1)
for ep in range(100):
    for (x,y) in zip(X,Y):
        pre_y=perceptron2(x)
        y=torch.Tensor([y])
        loss=nn.BCELoss()(pre_y,y)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

#读出感知机模型
t=list(perceptron2.parameters())
print(type(t)) #<class 'list'>
print(t) #[Parameter containing:tensor([[1.0317, 0.7094]], requires_grad=True), Parameter containing:tensor([-2.6672], requires_grad=True)]
print(t[0])#Parameter containing:tensor([[1.0317, 0.7094]], requires_grad=True)
print(t[0].data) #tensor([[1.0317, 0.7094]])
print(t[0].data[0])#tensor([1.0317, 0.7094])
w1,w2=t[0].data[0][0],t[0].data[0][1]
b=t[1].data[0]
s=f'学习的感知机为：y={w1.item()}*x1+{w2.item()}*x2+{b.item()}'
print(s)
