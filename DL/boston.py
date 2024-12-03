#波士顿房价预测
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

path=r"housing.data"
fp=open(path,'r',encoding='utf-8')
s=list(fp)
# print(s[0])
# s[0]=s[0].replace('\n','')
# print(s[0],type(s[0]))
# s[0]=s[0].split(' ')#返回一个列表，，但其中会有很多空格
# print(s[0],type(s[0]))

X,Y=[],[]
for i,line in enumerate(s):

    line=line.replace('\n','')
    line=line.split(' ')
    line2=[float(x) for x in line if x.strip()!='']
    X.append(line2[:-1])
    Y.append(line2[-1])
fp.close()
X=torch.FloatTensor(X)
Y=torch.FloatTensor(Y)
print(X.size(),Y.size())
print(len(X),X.size(0))
index=torch.randperm(X.size(0)) #打乱数据
print(type(index),index)
X=X[index]
Y=Y[index]
rate=0.8
train_size=int(X.size(0)*rate)
X_train,Y_train=X[:train_size],Y[:train_size]
X_test,Y_test=X[train_size:],Y[train_size:]
print(torch.min(X_train,dim=0)) #dim表示指定的维度,返回指定维度的最大数和对应下标
min=torch.min(X_train,dim=0)[0]
print(min,type(min))

#归一化函数
def map_minmax(data):
    min,max=torch.min(data,dim=0)[0],torch.max(data,dim=0)[0] #tensor([13])
    return (1.0*(data-min))/(max-min)

X_train=map_minmax(X_train)
X_test=map_minmax(X_test)
Y_train=map_minmax(Y_train)
Y_test=map_minmax(Y_test)

train_size=TensorDataset(X_train,Y_train)
train_loader=DataLoader(dataset=train_size,batch_size=16,shuffle=False)
test_size=TensorDataset(X_test,Y_test)
test_loader=DataLoader(dataset=test_size,batch_size=16,shuffle=False)

print(X_train.size(),Y_train.size())
del X,Y,X_train,Y_train,X_test,Y_test #释放内存

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(13,512)
        self.fc2=nn.Linear(512,1)
    def forward(self,x):
        pre_y=self.fc(x)
        pre_y=torch.sigmoid(pre_y)
        pre_y=self.fc2(pre_y)
        pre_y=torch.sigmoid(pre_y)
        return pre_y
model=model()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
ls=[]
print(train_loader)
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(f'Batch {batch_idx + 1}:')
#     print('Data:', data)
#     print('Target:', target)
#
#     # 如果只想打印前几个批次，可以添加条件
#     if batch_idx >= 0:  # 打印前5个batch
#         break
for ep in range(200):
    for i,(x,y) in enumerate(train_loader):
        pre_y=model(x)
        pre_y=pre_y.squeeze()
        loss=nn.MSELoss()(pre_y,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            ls.append(loss.item())

#测试
model.eval()
correct=0
lsy=torch.Tensor([])
ls=torch.Tensor([])
with torch.no_grad():
    for x,y in test_loader:
        pre_y=model(x)
        pre_y=pre_y.squeeze()
        t=(torch.abs(pre_y-y)<0.1)
        print(type(t))
        print(t) #tensor([ True,  True, False,  True,  True,  True, False,  True,  True,  True,False, False, False,  True, False, False])

        t=t.long().sum() #计算张量中非零元素的个数
        correct+=t
        ls=torch.cat((ls,pre_y))
        print('ls',ls)
        lsy=torch.cat((lsy,y))

print(f'正确率为：{100.*correct/len(test_loader.dataset)}')
plt.plot(ls)
plt.plot(lsy)
plt.show()
