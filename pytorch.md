# Pytorch学习笔记

## 一. Pytorch

### 1.1 概述

深度学习框架，简洁高效，易于上手，与python语法一致

### 1.2环境

miniconda：python + conda包管理，小型，够用！  https://docs.conda.io/en/latest/miniconda.html

anaconda：python + conda + 其他  https://www.anaconda.com/products/distribution/download-success-2

pytorch：https://pytorch.org/get-started/locally/

vc运行库：支持运行

pip源修改 :

```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

GPU环境

​	安装完anaconda或者miniconda

​	安装pytorch，如图选择CUDA版本

![image-20220530153643013](C:\Users\xiu\AppData\Roaming\Typora\typora-user-images\image-20220530153643013.png)

### 1.3 相关包安装

```
pip install pandas matplotlib notebook
数据分析结构化数据处理，绘图，开发环境（基于web开发）
```

### 1.4 notebook使用

```
控制台：jupyter notebook
```



## 二. 快速入门---简要介绍

### 2.1 tensor--张量

基于向量和矩阵的扩充，和numpy中的ndarray可以互相转换，共享共同的底层，无需要复制数据

![image-20220530211151222](C:\Users\xiu\AppData\Roaming\Typora\typora-user-images\image-20220530211151222.png)

#### 2.1.1 如何创建一个tensor

```python
import torch
import numpy as np
#------------------------------------------------#
#直接创建
t = torch.tensor([1,2,3])
#基于numpy创建
np_array = np.arange(12).reshape(3,4) 
t = torch.from_numpy(np_numpy)
```

#### 2.1.2 tensor的属性

```python
t.dtype
t.shape # 维度大小
t.size()
t.size(0) # 查看某一个维度的大小
```

#### 2.1.3 特殊tensor构建

```python
t = torch.zeros(3,4)
t = torch.ones(3,4)
x = torch.zeros_like(t) # 创建一个和t大小一样的全 0 tensor
x = torch.ones_like(t) # 全1，维度大小和t一样
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

#### 2.1.7 总结

![image-20220531154820468](C:\Users\xiu\AppData\Roaming\Typora\typora-user-images\image-20220531154820468.png)

### 2.2 神经网络





