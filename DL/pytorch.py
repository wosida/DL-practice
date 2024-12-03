import  torch

print(torch.__version__)
print(torch.cuda.is_available())

x0= torch.tensor(2) #0阶标量，torch.size为torch.size([]),写为（），ndim=0

x1= torch.tensor([2])#1阶向量，torch.size为torch.size([1]),写为（1） ndim=1
x2= torch.tensor([2,3,4])#1阶向量，torch.size为torch.size([3]),写为（3），ndim=1
x3= torch.tensor([[2,3,4],[5,6,7]])#2阶矩阵，torch.size为torch.size([2,3]),写为（2，3），ndim=2
# #一个feature map通常是三维的，对应一张彩色图，一个batch的特征图和彩色图像是四维的，视频是五维的（带有时间）
print(f'x0是{x0},维数为{x0.ndim},size为{x0.size()}')
print(f"x1是{x1},维数为{x1.ndim},size为{x1.size()}")
print(f"x2是{x2},维数为{x2.ndim},size为{x2.size()}")
print(f"x3是{x3},维数为{x3.ndim},size为{x3.size()}")

#生产内容随机，形状指定的张量
x4= torch.rand(32,3,224,224)#生成一个32*3*224*224的张量，即32个3通道224*224的张量
print(f"x4是{x4},维数为{x4.ndim},size为{x4.size()}") #维数为4，size为torch.size([32,3,224,224]),即一个batch的彩图，batch_size=32
#randn是标准正态分布，rand是0-1均匀分布
#randint是随机整数，randperm是随机排列
#ones是全1，zeros是全0，eye是单位矩阵
#arange是等差数列，linspace是等比数列
#full是指定值，cat是拼接，stack是堆叠
x= torch.tensor([1,2,3],dtype=torch.float32)
x= x.type(torch.float64)
print(x,x.dtype)
x=torch.tensor([1,2,3]).char()
print(x,x.dtype)


#张量的运算
x= torch.tensor([1,2,3])
y= torch.tensor([4,5,6])
print(x+y)#加法,jieguo tensor([5, 7, 9])
print(x-y)#减法,jieguo tensor([-3, -3, -3])
z=torch.add(x,y)#加法,jieguo tensor([5, 7, 9])  等效于z=x+y
print(z)
z= torch.sub(x,y)#减法,jieguo tensor([-3, -3, -3])等效于z=x-y
print(z)
z= torch.mul(x,y)#乘法,jieguo tensor([ 4, 10, 18])等效于z=x*y
print(z)
z= torch.div(x,y)#除法,jieguo tensor([0.2500, 0.4000, 0.5000])等效于z=x/y
print(z)
#orch.dot(input, tensor) → Tensor
#计算两个张量的点积（内积）
#官方提示：不能进行广播（broadcast）.
#example
torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1])) #即对应位置相乘再相加，结果为7
#torch.dot(torch.rand(2, 3), torch.rand(2, 2)) 报错，只允许一维的tensor
#RuntimeError: 1D tensors expected, got 2D, 2D tensors at /Users/distiller/project/conda/conda-bld/pytorch_1570710797334/work/aten/src/TH/generic/THTensorEvenMoreMath.cpp:774



#切片
x=torch.randint(0,10,[4,10])
print(x)  #x[::,3:8:2]  #取第1维所有，第2维开始切片,包左不包右
print(x[0])#取第一行


##数学函数，，第一维行向量，竖直方向；第二维列向量，水平方向
#sum函数
x=torch.tensor([[1,2,3],[4,5,6]],dtype=float)
print(x.sum())#tensor(21)
print(x.sum(dim=0))#tensor([5, 7, 9])
print(x.sum(dim=1))#tensor([ 6, 15])
#mean函数
print(x.mean())#tensor(3.5000)
print(x.mean(dim=0))#tensor([2.5000, 3.5000, 4.5000])
print(x.mean(dim=1))#tensor([2., 5.])
#max函数
print(x.max())#tensor(6)
print(x.max(dim=0))#torch.return_types.max(values=tensor([4, 5, 6]), indices=tensor([1, 1, 1]))
print(x.max(dim=1))#torch.return_types.max(values=tensor([3, 6]), indices=tensor([2, 2]))
#min函数，同max函数
#argmax函数，返回最大值的索引
print(x.argmax())#tensor(5)
print(x.argmax(dim=0))#tensor([1, 1, 1])
print(x.argmax(dim=1))#tensor([2, 2])
#argmin函数，同argmax函数
#to 方法
#x.to('cuda')#将张量转移到GPU上
#item方法,对于只有一个元素的张量，可以使用item方法将其转换为Python数值
#


#张量变形
x= torch.rand(4,4)
print(x)
x.reshape(2,8)
print(x)
print(x.shape)
print(x.size())   #shape和size都可以求形状，shape是属性，用中括号，size是方法，用小括号
#unsqueeze和squeeze   squeeze去掉长度为1的维度，unsqueeze升维，该维度长度为1
x.unsqueeze(1)# 在第一维升维
print(x.size())

#transpose()
x=torch.randint(0,6,[2,3])
y=x.transpose(0,1)#交换第一维和第二维
y=x.t()#t方法只适合二维矩阵，张量转置
y=x.permute([1,0]) ##对张量的维进行重新排列，相当于调换1和2维


##数学运算 1.基本的
#点积 ，dot（）函数 同等长度的一维向量相乘求和
#mm 传统意义上数学矩阵相乘，必须是2维矩阵
#bmm 带批量大小的矩阵相乘，同时对多张二维矩阵进行相乘，类似于三维矩阵相乘，第一维要相同
#matmul 多维矩阵相乘，后两维看作矩阵，，前几维要相同

#round(x)四舍五入
#exp(x) 指数
#log(x) 以e为底数
#log10(x) 10为底数
#pow(x,n) 幂运算 x的n次方


#梯度自动计算
#requires_grad,张量的一个属性，默认值为False，表示不能计算其梯度，若要计算需要设置为True
x=torch.tensor([3.],requires_grad=True)
y=torch.tensor([2.],requires_grad=True)
z=2*x**2-6*y**2
f=z**2
#f.backward()#自动求导
# print("f关于x的梯度是：",x.grad.item())
# print("f关于z的梯度是：",z.grad.item()) 中间变量的梯度计算不能这样，报错'NoneType' object has no attribute 'item'
def z_grad_save(g):  #因为即使是hook住的张量梯度，也会在运行结束后立刻释放掉，所以想要保存梯度，一定要在func函数里进行。
    global z_grad
    z_grad = g
    return None

z.register_hook(z_grad_save) #必须在自动求导以前注册hook
f.backward()
print("f关于z的梯度为：",z_grad.item())

##张量与其他对象之间转换
#1，numpy torch.from_numpy(a)   np.array(tensor) tensor.numpy()  tensor.tolist()
#PIL 使用transforms模块
import torchvision.transforms as transforms
## pil2tensor
# tfs=transforms.Compose([transforms.ToTensor()])(img) #在compose中添加操作才能对图片进行操作

#tensor2pil，直接用ToPILImage()(tensor)就可以对tensor进行操作

#拼接，只有在拼接的那一维上长度可以不同，其他必须相同
#x=torch.cat([x1,x2,x3],dim=1)

#简单模型



