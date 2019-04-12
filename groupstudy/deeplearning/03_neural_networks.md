## 第三章  多层全连接神经网络

　　深度学习的前身便是全连接神经网络，神经网络领域最开始主要是用来模拟人脑神经元系统，但是随后逐渐发展成了一项机器学习技术。多层全连接神经网络是现在深度学习各种网络的基础，了解它能够帮助我们更好地学其它内容。这一章从PyTorch入手，介绍PyTorch的处理对象、运算操作、自动求导，以及数据处理方法。接着从线性模型开始进入机器学习的内容，然后由Logistic回归引入分类问题，接着介绍全连接神经网络、反向传播算法、各种基于梯度的优化算法、数据预处理和训练技巧，最后用PyTorch实现多层全连接神经网络。

## 3.1 张量和梯度

　　本质上来说，PyTorch 是一个处理张量的库。一个张量是一个数字、向量、矩阵或任何 n 维数组。我们用单个数字创建一个张量：

```python
# Number
t1 = tourch.tensor(4.)
t1
```

结果为 tensor(4.)

4 . 是 4.0 的缩写。它用来表示你想创建浮点数的 Python（和 PyTorch）。我们可以通过检查张量的 dtype 属性来验证这一点：

```python
t1.dtype
```

torch.float32

我们可以试着创建复杂一点的张量：

```python
# Vector
t2 = torch.tensor([1., 2, 3, 4])
t2
```

tensor([1., 2., 3., 4.])

```python
# Matrix
t3 = torch.tensor([[5., 6],[7, 8], [9, 10]])
t3
```

tensor([[5., 6.],
              [7., 8.],
              [9., 10.]])

```python
# 3-dimensional array
t4 = torch.tensor([
    [[11, 12, 13],
     [13, 14, 15]],
    [[15, 16, 17],
     [17, 18, 19]]])
t4
```

tensor([[[11., 12., 13.],[13., 14., 15.]],[[15., 16., 17.],[17., 18., 19.]]])
张量可以有任何维数。每个维度有不同的长度。我们可以用张量的.shape 属性来查看每个维度的长度。

```python
t1.shape
```

torch.Size([])

```python
t2.shape
```

torch.Size([4])

```python
t3.shape
```

torch.Size([3, 2])

```python
t4.shape
```

torch.Size([2, 2, 3])

我们可以将张量与常用的算数运算相结合。如下：

```python
# Create tensors
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
```

我们已经创建了 3 个张量：x、w 和 b。w 和 b 有额外的参数 requires_grad，设置为 True。一会儿就可以看看它能做什么。

通过结合这些张量，我们可以创建新的张量 y。

```python
# Arithmetic operations
y = w * x + b
y
```

tensor(17., grad_fn=<AddBackward0>)

如预期所料，y 是值为 3 * 4 + 5 = 17 的张量。PyTorch 的特殊之处在于，我们可以自动计算 y 相对于张量（requires_grad 设置为 True）的导数，即 w 和 b。为了计算导数，我们可以在结果 y 上调用.backward 方法。

```python
# Compute derivatives
y.backward()
```

y 相对于输入张量的导数被存储在对相应张量的.grad 属性中。

```python
# Display gradients
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)
```

dy/dx:  None

dy/dw: tensor(3.)

dy/db: tensor(1.)

如预期所料，dy/dw 的值与 x 相同（即 3），dy/db 的值为 1。注意，x.grad 的值为 None，因为 x 没有将 requires_grad 设为 True。w_grad 中的「grad」代表梯度，梯度是导数的另一个术语，主要用于处理矩阵。

**与 Numpy 之间的互操作性**

Numpy 是 Python 中用于数学和科学计算的流行开源库。它支持在大型多维数组上进行高效运算，拥有一个支持多个库的大型生态系统。这些库包括：

- 用于画图、可视化的 Matplotlib
- 用于图像和视频处理的 OpenCV
- 用于文件 I/O 和数据分析的 Pandas

PyTorch 并没有重新创造 wheel，而是与 Numpy 很好地交互，以利用它现有的工具和库生态系统。

```python
import numpy as np
x = np.array([[1, 2], [3, 4]])
```

可以用 torch.fron_numpy 将 Numpy 数组转化为 PyTorch 张量。

```python
# Convert the numpy array to a torch tensor
y = torch.from_numpy(x)
```

接下来可以验证 Numpy 数组和 PyTorch 张量是否拥有类似的数据类型。

```python
x.dtype, y.dtype
```

(dtype('int64'), torch.int64)

可以使用张量的.to_numpy 方法将 PyTorch 张量转化为 Numpy 数组。

```python
# Conver a torch tensor to a numpy array
z = y.numpy()
```

PyTorch 和 Numpy 之间的互操作性真的非常重要，因为你要用的大部分数据集都可能被读取并预处理为 Numpy 数组。

**延伸阅读**

PyTorch 中的张量支持很多运算，这里列出的并不详尽。如果你想了解更多关于张量和张量运算的信息，可参考以下地址：

链接：https://pytorch.org/docs/stable/tensors.html

## 3.2 线性回归

我们将讨论机器学习的一大基本算法：线性回归。我们将创建一个模型，使其能根据一个区域的平均温度、降雨量和湿度（输入变量或特征）预测苹果和橙子的作物产量（目标变量）。训练数据如下：

![](http://ww1.sinaimg.cn/mw690/6deb72a3ly1g1znff14upj20iy054gno.jpg)

在线性回归模型中，每个目标变量的估计方式都是作为输入变量的一个加权和，另外还会有某个常量偏移（也被称为偏置量）：

yield_apple = w11 * temp + w12 * rainfall + w13 * humidity + b1

yield_orange = w21 * temp + w22 * rainfall + w23 * humidity + b2 

可视化地看，这意味着苹果或橙子的产量是温度、降雨量或湿度的线性函数或平面函数：

![](http://ww1.sinaimg.cn/mw690/6deb72a3ly1g1zng3qumej20gk0aedkc.jpg)

*因为我们只能展示三个维度，所以此处没有给出湿度*

线性回归的「学习」部分是指通过检视训练数据找到一组权重（w11、w12…w23）和偏置 b1 和 b2），从而能根据新数据得到准确的预测结果（即使用一个新地区的平均温度、降雨量和湿度预测苹果和橙子的产量）。为了得到更好的结果，这个过程会对权重进行许多次调整，其中会用到一种名为「梯度下降」的优化技术。首先我们导入 Numpy 和 PyTorch：

```python
import numpy as np
import torch
```

训练数据可以使用两个矩阵表示：输入矩阵和目标矩阵；其中每个矩阵的每一行都表示一个观察结果，每一列都表示一个变量。

```python
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')
# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')
```

我们已经分开了输入变量和目标变量，因为我们将分别操作它们。另外，我们创建的是 numpy 数组，因为这是常用的操作训练数据的方式：将某些 CSV 文件读取成 numpy 数组，进行一些处理，然后再将它们转换成 PyTorch 张量，如下所示：

```python
# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)
```

权重和偏置（w11、w12…w23、b1 和 b2）也可表示成矩阵，并初始化为随机值。w 的第一行和 b 的第一个元素用于预测第一个目标变量，即苹果的产量；对应的第二个则用于预测橙子的产量。

```python
# Weights and biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print(w)
print(b)
```

torch.randn 会创建一个给定形状的张量，其中的元素随机选取自一个均值为 0 且标准差为 1 的正态分布。该模型实际上就是一个简单的函数：执行输入 x 和权重 w 的矩阵乘法，再加上偏置 b（每个观察都会重复该计算）。

![](http://ww1.sinaimg.cn/mw690/6deb72a3ly1g1znj7odsxj20iz06hmxp.jpg)

我们可将该模型定义为：

```python
def model(x):
    return x @ w.t() + b
```

@ 表示 PyTorch 中的矩阵乘法，.t 方法会返回一个张量的转置。通过将输入数据传入模型而得到的矩阵是目标变量的一组预测结果。

```python
# Generate predictions
preds = model(inputs)
print(preds)
```

tensor([[-70.6641, 253.4470],
        [-96.9173, 336.8742],
        [-74.1960, 435.6436],
        [-91.2382, 213.2355],
        [-85.8809, 344.9089]], grad_fn=<AddBackward0>)

接下来比较我们的模型的预测结果与实际的目标。

```python
# Compare with targets
print(targets)
```

tensor([[ 56.,  70.],
        [ 81., 101.],
        [119., 133.],
        [ 22.,  37.],
        [103., 119.]])

可以看到，我们的模型的预测结果与目标变量的实际值之间差距巨大。很显然，这是由于我们的模型的初始化使用了随机权重和偏置，我们可不能期望这些随机值就刚好合适。

**损失函数**

在我们改进我们的模型之前，我们需要一种评估模型表现优劣的方法。我们可以使用以下方法比较模型预测和实际目标：

- 计算两个矩阵（preds 和 targets）之间的差异；
- 求这个差异矩阵的所有元素的平方以消除其中的负值；
- 计算所得矩阵中元素的平均值。

结果是一个数值，称为均方误差（MSE）。

```python
# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()
```

torch.sum 返回一个张量中所有元素的和，.numel 方法则返回一个张量中元素的数量。我们来计算一下我们模型的当前预测的均方误差：

```python
# Compute loss
loss = mse(preds, targets)
print(loss)
```

tensor(39649.8359, grad_fn=<DivBackward0>)

我们解读一下这个结果：平均而言，预测结果中每个元素与实际目标之间的差距大约为 215（46194 的平方根）。考虑到我们所要预测的数值的范围本身只有 50-200，所以这个结果实在相当糟糕。我们称这个结果为损失（loss），因为它指示了模型在预测目标变量方面的糟糕程度。损失越低，模型越好。

**计算梯度**

使用 PyTorch，我们可以根据权重和偏置自动计算 loss 的梯度和导数，因为它们已将 requires_grad 设置为 True。

```python
# Compute gradients
loss.backward()
```

这些梯度存储在各自张量的 .grad 属性中。注意，根据权重矩阵求得的 loss 的导数本身也是一个矩阵，且具有相同的维度。

```python
# Gradients for weights
print(w)
print(w.grad)
```

tensor([[-0.6603,  0.3072, -0.9914],
        [ 0.7547,  2.3412,  0.9846]], requires_grad=True)
tensor([[-13365.6172, -14606.6602,  -9090.0225],
        [ 18949.9844,  20573.5020,  12574.3672]])

这个损失是我们的权重和偏置的一个二次函数，而我们的目标是找到能使得损失最低的权重集。如果我们绘制出任意单个权重或偏置元素下的损失的图表，我们会得到类似下图的结果。通过微积分，我们可以了解到梯度表示损失的变化率，即与权重和偏置相关的损失函数的斜率。

如果梯度元素为正数，则：

- 稍微增大元素的值会增大损失。

- 稍微减小元素的值会降低损失。

![](http://ww1.sinaimg.cn/large/6deb72a3ly1g1zpzx8dx9j20t30avgni.jpg)

作为权重的函数的 MSE 损失（蓝线表示梯度）

如果梯度元素为负数，则：

- 稍微增大元素的值会降低损失。

- 稍微减小元素的值会增大损失。

![](http://ww1.sinaimg.cn/large/6deb72a3ly1g1zq0g75h4j20t10auabw.jpg)

作为权重的函数的 MSE 损失（绿线表示梯度）

通过改变一个权重元素而造成的损失的增减正比于该元素的损失的梯度值。这就是我们用来提升我们的模型的优化算法的基础。

在我们继续之前，我们通过调用 .zero() 方法将梯度重置为零。我们需要这么做的原因是 PyTorch 会累积梯度，也就是说，我们下一次在损失上调用 .backward 时，新的梯度值会被加到已有的梯度值上，这可能会导致意外结果出现。

```python
w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)
```

tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([0., 0.])

**使用梯度下降调整权重和偏置**

我们将使用梯度下降优化算法来降低损失和改善我们的模型，步骤如下：

1. 生成预测
2. 计算损失
3. 根据权重和偏置计算梯度
4. 按比例减去少量梯度来调整权重
5. 将梯度重置为零

下面我们一步步地实现：

```python
# Generate predictions
preds = model(inputs)
print(preds)
```

tensor([[-70.6641, 253.4470],
        [-96.9173, 336.8742],
        [-74.1960, 435.6436],
        [-91.2382, 213.2355],
        [-85.8809, 344.9089]], grad_fn=<AddBackward0>)

注意，这里的预测结果和之前的一样，因为我们还未对我们的模型做出任何修改。损失和梯度也是如此。

```python
# Calculate the loss
loss = mse(preds, targets)
print(loss)
```

tensor([[-13365.6172, -14606.6602,  -9090.0225],
        [ 18949.9844,  20573.5020,  12574.3672]])
tensor([-159.9793,  224.8219])

最后，使用上面计算得到的梯度更新权重和偏置。

```python
# Adjust weights & reset gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()
```

上面需要注意的几点：

- 我们使用 torch.no_grad 指示 PyTorch 我们在更新权重和偏置时不应该跟踪、计算或修改梯度。
- 我们为梯度乘上了一个非常小的数值（这个案例中为 10^-5），以确保我们不会改变权重太多，因为我们只想在梯度的下降方向上迈出一小步。这个数值是这个算法的学习率（learning rate）。
- 在更新权重之后，我们将梯度重置为零，以免影响后续计算。

现在我们来看看新的权重和偏置：

```python
print(w)
print(b)
```

tensor([[-0.5267,  0.4533, -0.9005],
        [ 0.5652,  2.1355,  0.8589]], requires_grad=True)
tensor([-0.4135, -0.8476], requires_grad=True)

使用新的权重和偏置，模型的损失应更低。

```python
# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)
```

tensor(26765.1836, grad_fn=<DivBackward0>)

只是简单地使用梯度下降来稍微调整权重和偏置，我们就已经实现了损失的显著下降。

**多次训练**

为了进一步降低损失，我们可以多次使用梯度重复调整权重和偏置的过程。一次迭代被称为一个 epoch。我们训练模型 100 epoch 看看。

```python
# Train for 100 epochs
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()
```

再次验证，现在损失应该会更低：

```python
# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)
```

tensor(95.0308, grad_fn=<DivBackward0>)

可以看到，现在的损失比我们开始时低了很多。我们看看模型的预测结果，并将其与目标比较一下。

```python
# Predictions
preds
```

tensor([[ 59.2217,  69.7643],
        [ 75.2371,  95.5674],
        [131.2372, 145.4368],
        [ 32.5877,  34.2057],
        [ 83.0004, 111.7312]], grad_fn=<AddBackward0>)

```python
# Targets
targets
```

```
tensor([[ 56.,  70.],
        [ 81., 101.],
        [119., 133.],
        [ 22.,  37.],
        [103., 119.]])
```

现在的预测结果已非常接近目标变量；而且通过训练模型更多 epoch，我们还能得到甚至更好的结果。

**使用 PyTorch 内置的线性回归**

上面的模型和训练过程是使用基本的矩阵运算实现的。但因为这是一种非常常用的模式，所以 PyTorch 配备了几个内置函数和类，让人能很轻松地创建和训练模型。

首先从 PyTorch 导入 torch.nn 软件包，其中包含了用于创建神经网络的效用程序类。

```python
import torch.nn as nn
```

和之前一样，我们将输入和目标表示成矩阵形式。

```python
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], 
                    [22, 37], [103, 119], [56, 70], 
                    [81, 101], [119, 133], [22, 37], 
                    [103, 119], [56, 70], [81, 101], 
                    [119, 133], [22, 37], [103, 119]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
```

我们这一次使用 15 个训练样本，以演示如何以小批量的形式处理大数据集。

**数据集和数据加载器**

我们将创建一个 TensorDataset，这让我们可以读取 inputs 和 targets 的行作为元组，并提供了 PyTorch 中用于处理许多不同类型的数据集的标准 API。

```python
from torch.utils.data import TensorDataset

# Define dataset
train_ds = TensorDataset(inputs, targets)
train_ds[0:3]
```

(tensor([[ 73.,  67.,  43.],
         [ 91.,  88.,  64.],
         [ 87., 134.,  58.]]), tensor([[ 56.,  70.],
         [ 81., 101.],
         [119., 133.]]))

TensorDataset 让我们可以使用数组索引表示法（上面代码中的 [0:3]）读取一小部分训练数据。它会返回一个元组（或配对），其中第一个元素包含所选行的输入变量，第二个元素包含目标，

我们还将创建一个 DataLoader，它可以在训练时将数据分成预定义大小的批次。它还能提供其它效用程序，如数据的混洗和随机采样。

```python
from torch.utils.data import DataLoader

# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
```

数据加载器通常搭配 for-in 循环使用。举个例子：

```python
for xb, yb in train_dl:
    print(xb)
    print(yb)
    break
```

tensor([[ 87., 134.,  58.],
        [102.,  43.,  37.],
        [ 69.,  96.,  70.],
        [102.,  43.,  37.],
        [ 69.,  96.,  70.]])
tensor([[119., 133.],
        [ 22.,  37.],
        [103., 119.],
        [ 22.,  37.],
        [103., 119.]])

在每次迭代中，数据加载器都会返回一批给定批大小的数据。如果 shuffle 设为 True，则在创建批之前会对训练数据进行混洗。混洗能帮助优化算法的输入随机化，这能实现损失的更快下降。

**nn.Linear**

除了人工地实现权重和偏置的初始化，我们还可以使用 PyTorch 的 nn.Linear 类来定义模型，这能自动完成初始化。

```python
# Define model
model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)
```

Parameter containing:
tensor([[-0.3113,  0.3951,  0.2852],
[-0.0276, -0.5176,  0.4246]], requires_grad=True)
Parameter containing:
tensor([-0.0141,  0.1626], requires_grad=True)

PyTorch 模型还有一个很有用的 .parameters 方法，这能返回一个列表，其中包含了模型中所有的权重和偏置矩阵。对于我们的线性回归模型，我们有一个权重矩阵和一个偏置矩阵。

```python
# Parameters
list(model.parameters())
```

[Parameter containing:
 tensor([[-0.3113,  0.3951,  0.2852],
         [-0.0276, -0.5176,  0.4246]], requires_grad=True),
 Parameter containing:
 tensor([-0.0141,  0.1626], requires_grad=True)]

我们可以使用之前一样的方式来得到模型的预测结果：

```python
# Generate predictions
preds = model(inputs)
preds
```

tensor([[ 15.9968, -18.2742],
        [ 24.6793, -20.7239],
        [ 42.3887, -46.9722],
        [ -4.2238,  -9.1995],
        [ 36.3992, -21.7098],
        [ 15.9968, -18.2742],
        [ 24.6793, -20.7239],
        [ 42.3887, -46.9722],
        [ -4.2238,  -9.1995],
        [ 36.3992, -21.7098],
        [ 15.9968, -18.2742],
        [ 24.6793, -20.7239],
        [ 42.3887, -46.9722],
        [ -4.2238,  -9.1995],
        [ 36.3992, -21.7098]], grad_fn=<AddmmBackward>)

**损失函数**

除了手动定义损失函数，我们也可使用内置的损失函数 mse_loss：

```python
# Import nn.functional
import torch.nn.functional as F
```

nn.functional 软件包包含很多有用的损失函数和其它几个效用程序。

```python
# Define loss function
loss_fn = F.mse_loss
```

我们计算一下我们模型的当前预测的损失。

```python
loss = loss_fun(model(inputs), targets)
print(loss)
```

tensor(9269.7607, grad_fn=<MseLossBackward>)

**优化器**

除了以人工方式使用梯度操作模型的权重和偏置，我们也可使用优化器 optim.SGD。SGD 表示「随机梯度下降」。之所以是「随机」，原因是样本是以批的形式选择（通常会用到随机混洗），而不是作为单独一个数据组。

```python
# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)
```

注意，这里的 model.parameters() 是 optim.SGD 的一个参数，这样优化器才知道在训练步骤中应该修改哪些矩阵。另外，我们还可以指定一个学习率来控制参数每次的修改量。

**训练模型**

我们现在已经准备好训练模型了。我们将遵循实现梯度下降的同一过程：

- 生成预测
- 计算损失
- 根据权重和偏置计算梯度
- 按比例减去少量梯度来调整权重
- 将梯度重置为零

唯一变化的是我们操作的是分批的数据，而不是在每次迭代中都处理整个训练数据集。我们定义一个效用函数 fit，可训练模型给定的 epoch 数量。

```python
# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            preds = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(preds, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
```

上面需要注意的几点：

- 我们使用之前定义的数据加载器来为每个迭代获取数据批次。
- 我们没有手动更新参数（权重和偏置），而是使用了 opt.step 来执行更新，并使用了 opt.zero_grad 来将梯度重置为零。
- 我们还添加了一个日志语句，能够显示每第 10 个 epoch 的最后一批数据的损失，从而可让我们跟踪训练进程。loss.item 会返回存储在损失张量中的实际值。

训练模型 100 epoch。

```python
fit(100, model, loss_fn, opt)
```

Epoch [10/100], Loss: 1.3765
Epoch [20/100], Loss: 1.1817
Epoch [30/100], Loss: 2.9769
Epoch [40/100], Loss: 1.3210
Epoch [50/100], Loss: 1.1224
Epoch [60/100], Loss: 0.3655
Epoch [70/100], Loss: 0.5386
Epoch [80/100], Loss: 0.4578
Epoch [90/100], Loss: 1.6593
Epoch [100/100], Loss: 0.9839

接下来使用我们的模型生成预测结果，再验证它们与目标的接近程度。

```python
# Generate predictions
preds = model(inputs)
preds
```

tensor([[ 58.3422,  70.9988],
        [ 81.3429, 101.1752],
        [118.5515, 130.6639],
        [ 28.4375,  41.5285],
        [ 96.0359, 117.2896],
        [ 58.3422,  70.9988],
        [ 81.3429, 101.1752],
        [118.5515, 130.6639],
        [ 28.4375,  41.5285],
        [ 96.0359, 117.2896],
        [ 58.3422,  70.9988],
        [ 81.3429, 101.1752],
        [118.5515, 130.6639],
        [ 28.4375,  41.5285],
        [ 96.0359, 117.2896]], grad_fn=<AddmmBackward>)

```python
# Compute with targets
targets
```

tensor([[ 56.,  70.],
        [ 81., 101.],
        [119., 133.],
        [ 22.,  37.],
        [103., 119.],
        [ 56.,  70.],
        [ 81., 101.],
        [119., 133.],
        [ 22.,  37.],
        [103., 119.],
        [ 56.,  70.],
        [ 81., 101.],
        [119., 133.],
        [ 22.,  37.],
        [103., 119.]])

实际上，这个预测结果非常接近我们的目标。现在，我们有了一个相当好的预测模型，可以根据一个地区的平均温度、降雨量和湿度预测苹果和橙子的产量。

## 4.3 使用 logistic 回归实现图像分类

