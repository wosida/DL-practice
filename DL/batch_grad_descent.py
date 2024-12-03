#一个样本就有一个对应的目标函数，对于一个样本，我们可以计算出目标函数的梯度，然后根据梯度更新参数，这就是随机梯度下降法。
#但是，如果我们有多个样本，我们可以计算出所有样本的目标函数的梯度，然后求平均，这就是批量梯度下降法。
import torch
torch.manual_seed(123)
X=[1,2,3,4,5,6,7,8,9,10]
Y=[2,3,4,5,6,7,8,9,10,11]
X=torch.tensor(X,dtype=float)
Y=(torch.tensor(Y,dtype=float))
#数据打包
n=4
X1,Y1=X[:n],Y[:n]
X2,Y2=X[n:2*n],Y[n:2*n]
X3,Y3=X[2*n:3*n],Y[2*n:3*n]
X,Y=[X1,X2,X3],[Y1,Y2,Y3]
print(X,Y) # [tensor([1., 2., 3., 4.], dtype=torch.float64), tensor([5., 6., 7., 8.], dtype=torch.float64), tensor([ 9., 10.], dtype=torch.float64)]
print(type(X)) #<class 'list'>
def f(x):
    return w*x+b
def dw(x,y):
    return x*(f(x)-y)
def db(x,y):
    return f(x)-y
w=torch.randn(1,requires_grad=True)
b=torch.randn(1,requires_grad=True)
lr=0.01
Z=zip(X,Y)
print(Z) #<zip object at 0x7f2f8b3b3f40>
for x,y in Z:
    print(x,y)
    print(type(x),type(y))
    dw_value=dw(x,y)
    db_value=db(x,y)
    print(dw_value,db_value)