# Pytorch学习笔记

## 一. Pytorch

### 1.1 概述

深度学习框架，简洁高效，易于上手，与python语法一致

### 1.2 环境

miniconda：python + conda包管理，小型，够用！  https://docs.conda.io/en/latest/miniconda.html

anaconda：python + conda + 其他  https://www.anaconda.com/products/distribution/download-success-2

pytorch：https://pytorch.org/get-started/locally/

vc运行库：支持运行

清华大学：

```
https://mirrors.tuna.tsinghua.edu.cn/
```

pip源修改 :

```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

GPU环境

​	安装完anaconda或者miniconda

​	安装pytorch：

```
pytorch 官网下载
```



删除pytorch

```
conda uninstall pytorch
conda uninstall libtorch
```

conda源修改: （直接修改用户目录下的配置文件）

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge

conda config --set show_channel_urls yes
```



### 1.3 相关包安装

```
pip install pandas matplotlib notebook
数据分析结构化数据处理，绘图，开发环境（基于web开发）
```

### 1.4 notebook使用

```
控制台：jupyter notebook
```

### 1.5 本地配置云主机GPU环境

```
AuToDL：pycharm配置运行环境即可，类似于虚拟环境的配置（anaconda）
文件上传：FileZilla
```

### 1.6 网络GPU

```
Kaggle notebook
阿里云天池
```



## 二. 快速入门

### 2.1 Tensor

向量和矩阵的**扩充**，张量可以取描述任何维度的事物！

包括：类型，创建，属性，运算，操作，类型转换

#### 2.1.1 创建

```python
import torch
import numpy as np
t = torch.tensor([1,2,3]) # 直接创建

np_array = np.arange(12).reshape(3,4) # 基于numpy创建
t = torch.from_numpy(np_numpy) # 转换

torch.rand(1,1) # 0-1之间
torch.randn(1,1) # 标准正态分布的伪随机数（均值为0，方差为1）
torch.Tensor(2,2).uniform(-1,1) # -1 ~ 1均匀分布

t = torch.zeros(3,4) # 3 * 4 全 0
t = torch.ones(3,4) # 3 * 4 全 1

x = torch.zeros_like(t) # 创建一个和t大小一样的全 0 tensor
x = torch.ones_like(t) # 全1，维度大小和t一样

a = linspace(0, 10, 4) # 取4个，0 - 10 等 间隔，包括 10
a = torch.arange(0,10,1) # 0 - 10, 步长是1，不包括 10  LongTensor
a = torch.randperm(10) # 0 - 9 排列随机的数 LongTensor
```

---

```python
# 对比numpy
import numpy as np
a = np.array([[1,2],[3,4]])
zeros
ones
```



#### 2.1.2 属性

```python
# t.dtype，device，layout

t.shape # 属性，维度大小
t.size() # 方法，获取维度
t.size(0) # 方法，获取某一个维度的大小
```

#### 2.1.3 运算

```python

```

#### 2.1.4 其他

```python
t.device # tensor的位置
torch.cuda.is_available() # GPU可用？
t.to('cuda') # 将tensor移到显存 
t = t.cuda() # 另一种写法
```

#### 2.1.5 tensor的运算

```python
t.dtype # 查看类型
t = t.type(torch.int64) # 类型转换
t = t.int()

# 加法
t1 = torch.ones(3,3)
t1 + 1
t1 + t1
t1.add(1) #tensor([[1., 1., 1.],
          #[1., 1., 1.],
          #[1., 1., 1.]])
t1.add_(1) # 改变了，全部是2


# 求abs mean max min T
torch.abs(t1)
mutual(t1,t1.T) # 矩阵乘法
t@(t.T) # 简写矩阵乘法

t1.max() # tensor类型
t1.min().item() # 转化为python 浮点数


# numpy -> tensor
t = torch.form_numpy(np_a)
# tensor -> numpy
t.numpy()

# view 变形
t = torch.ones(3,4)
t1 = t.view(12,1) # 变成12*1
t2 = t.view(-1,1) # 变成一列

# squeeze 去掉维度是1的维度
```

#### 2.1.6 自动微分--Autograd

```python
t.requires_grad # 是否跟踪运算
tt = torch.ones(3,3,requires_grad = True) # 跟踪
y = tt + 1
y.grad_fn # 输出add，表示tensor是由add运算来的


out = y.mean() # 得到标量，平均值
out.backward() # 反向传播
tt.grad # 计算梯度
tt.grad.data.zero_() # 梯度清0

#-------------------------------#
# 不跟踪，也就是想打印某些值的时候
with torch.no_grad():
    y = t_sum + 1
    print(y.requires_grad) # False
#-------------------------------#


#-------------------------------#
# 截断跟踪
out = y.detach() #之后运用out的运算是不被跟踪的，表示到此为止是截断
#-------------------------------#

```



### 2.2 神经网络

#### 2.2.1 激活函数

激活函数为神经网络带来了非线性，可以解决非线性拟合问题，增强拟合度。

常见的有：

relu：f(x) = max(0,x)

```
torch.relu(input)
```

sigmoid：映射到0-1之间
```
torch.sigmoid(input)
```
tanh：-1到1之间

```
torch.tanh(input)
```

### 2.3 多分类问题与通用训练函数

####  2.3.1 softmax分类 

- 多分类问题
- 每个样本都必须属于某一个类别，所有可能的样本都被覆盖
- 得到属于每一个类别的概念值
- 所有的概率相加为1

```python
# 计算交叉熵
nn.CrossEntropyLoss() 
```

#### 2.3.2 MNIST手写识别

- 提供常用额数据集、模型、转换函数等
- 内置数据集用于测试学习和创建基准模型

统一数据加载和处理类

```python
torch.utils.data.Dataset
torch.utils.data.DataLoader
# 代码模块化
```

MNIST手写识别(除了课程之外，应当参考论文进行学习（待）)

```python
import torch
import torchvision as tv
from torchvision.transforms import ToTensor
# datasets 创建
train_ds = tv.datasets.MNIST('data',train = True,transform = ToTensor(),download = True)
test_ds = tv.datasets.MNIST('data',train = False,transform = ToTensor(),download = True)
# ToTensor() (h,w,c) -> (c,h,w)

# dataloader 创建
train_dl = torch.utils.data.DataLoader(train_ds, batch_size = 32, shuffle = True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size = 32)
# torch.utils.data.DataLoader(datasets, batch_size,suffle) 数据集，一次迭代的大小，乱序

imgs,labels = next(iter(train_dl))
print(imgs.shape) # 查看图片的一个批次（batch_size）的维度


# 模型创建
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义模型的各个层次
        self.linear_1 = nn.Linear(28*28,128)
        self.linear_2 = nn.Linear(128,64)
        self.linear_3 = nn.Linear(64,10)
    def forward(self,x): 
        # 定义处理过程
        x = x.view(-1,1*28*28)
        x = torch.relu(self.linear_1(x))
        x = torch.relu(self.linear_2(x))
        logits = self.linear_3(x) # 不进行激活，原因：torch.nn.CrossEntropyLoss()，计算CELoss需要的输入就是logits
        return logits   

# 训练一次的函数
def train(dl,model,loss_fn,optimizer,device):
    size = len(dl.dataset) # 图片的数量
    # 需要多少步
    iter = len(dl) # 在dl中设置了batch_size的大小，自动可以得出步
    total_loss,total_acc = 0,0 #累加loss，预测正确的样本数
    for x, y in dl: # 得到输入以及输出
        x,y = x.to(device),y.to(device)
        pred = model(x)
        loss = loss_fn(pred,y) # 预测，真实

        optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播
        optimizer.step() # 梯度下降
        
        with torch.no_grad(): # 不跟踪梯度
            total_acc += (pred.argmax(1) == y).type(torch.float).sum().item() # 累加计算正确的样本
            total_loss += loss.item() # 累加批次loss
    train_acc = total_acc / size
    train_loss = total_loss / iter # 批次的平均loss
    return train_acc,train_loss

# 定义测试函数
def test(dl,model,loss_fn,device):
    size = len(dl.dataset) # 数量
    iter = len(dl) # 多少批次
    total_loss,total_acc = 0,0 #累加loss，预测正确的样本数
    for x, y in dl: # 得到输入以及输出
        x,y = x.to(device),y.to(device)
        pred = model(x)
        loss = loss_fn(pred,y) # 预测，真实
        
        total_acc += (pred.argmax(1) == y).type(torch.float).sum().item() # 累加计算正确的样本
        total_loss += loss.item() # 累加批次loss
    test_acc = total_acc / size
    test_loss = total_loss / iter # 批次的平均loss
    return test_acc,test_loss

# 封装模型训练函数
def fit(epochs,model,train_dl,test_dl,loss_fn,optimizer,device):
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    for epoch in range(epochs):
        e_test_acc,e_test_loss = test(test_dl,model,loss_fn,device)
        e_train_acc,e_train_loss = train(train_dl,model,loss_fn,optimizer,device)
        train_acc.append(e_train_acc)
        train_loss.append(e_train_loss)
        
        test_acc.append(e_test_acc)
        test_loss.append(e_test_loss)
        template = 'epoch:{:2d}, train_acc:{:.2f}, train_loss:{:.2f}, test_acc:{:.2f}, test_loss:{:.2f}'
        print(template.format(epoch,e_train_acc,e_train_loss,e_test_acc,e_test_loss))
    return train_acc,train_loss,test_acc,test_loss

# 损失函数（loss_fn），模型初始化（model），模型参数（parameters），学习率（lr），优化器选择（optimizer）
loss_fn = torch.nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model().to(device)
parameters = model.parameters() # ()获取模型参数的方法
lr = 0.001 # 学习率
optimizer = torch.optim.SGD(parameters,lr) # 优化器，参数和lr
epochs = 10
# device gpu
train_acc = []
train_loss = []
test_acc = []
test_loss = []
train_acc,train_loss,test_acc,test_loss = fit(epochs,model,train_dl,test_dl,loss_fn,optimizer,device)
```



### 2.4 深度学习python绘图

```python
import matplotlib.pyplot as plt
import numpy as np
plt.plot(range(epochs), train_acc, label = 'train loss') # x,y
plt.plot(range(epochs), test_acc, label = 'test loss') # x,y
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend() # 添加图例，label才能生效
plt.savafig('loss.png') # 图像的保存
```

### 2.5 基础部分总结

```

```



## 三. 计算机视觉基础

### 3.1 CNN

卷积是对图像特征进行提取，并输出结果

不同的卷积核对于图像特征的敏感度是不一样的

### 3.2 CNN架构

```
卷积层：conv2d
非线性层（激活层）：sigmiod / tanh / relu
池化层：pooling2d
全连接层：wx + b
```

