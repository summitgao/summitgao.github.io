## 第三章  基于PCA和k-means的SAR图像变化检测

　　介绍的第一个算法是基于PCA和k-means的SAR图像变化检测算法，由T. Celik于2009年发表于IEEE Geoscience and Remote Sensing Letters，该算法提供了公开代码，目前Google Scholar被引用次数达到384次（在遥感图像处理领域，引用率算非常高了）。

![](http://ww1.sinaimg.cn/large/6deb72a3ly1g1yaej241qj20t904o74f.jpg)

论文代码下载地址为：https://yunpan.360.cn/surl_yFzvh3New75 （提取码：ddc7）

## 2.2.1 主成分分析算法（PCA）

　　在一组多变量的数据中，很多变量常常是一起变动的。一个原因是很多变量是同一个驱动影响的的结果。在很多系统中，只有少数几个这样的驱动，但是多余的仪器使我们测量了很多的系统变量。当这种情况发生的时候，你需要去掉信息中的冗作，可以通过用一个简单的新变量代替这组变量来简化此问题。

　　主成分分析（principal component analysis, PCA）是最重要的降维算法之一，也是解决上面问题的最常用思路。PCA 的思想是将维特征映射到维上，这维是全新的正交特征。这维特征称为主元，是重新构造出来的维特征。在 PCA 中，数据从原来的坐标系转换到新的坐标系下，新的坐标系的选择与数据本身是密切相关的。其中，第一个新坐标轴选择的是原始数据中方差最大的方向，第二个新坐标轴选取的是与第一个坐标轴正交且具有最大方差的方向，依次类推，我们可以取到好多个这样的坐标轴。

　　PCA的基本原理就是如何，但具体细节上如何操作的，可以百度搜索一篇文章《主成分分析-最大方差解释》，这个里面写的比较详细。**其实对我来说，PCA的算法流程是怎样的，已经不记得了，我只知道它是用来降维的。**大家掌握到这个程度，也就可以了，具体算法细节有时间可以看看，没时间看影响也不大。

## 2.2.2. K 均值聚类

　　k-均值聚类（k-means）是一种简单的迭代型聚类算法，采用距离作为相似性指标，从而发现给定数据集中的K个类，且每个类的中心是根据类中所有值的均值得到，每个类用聚类中心来描述。对于给定的一个包含$n​$个$d​$维数据点的数据集$X​$以及要分得的类别$k​$,选取欧式距离作为相似度指标，聚类目标是使得各类的聚类平方和最小，即最小化：

　　$$J=\sum^K_{k=1}\sum^n_{i=1}\lVert  x_i-u_k \rVert^2$$

　　结合最小二乘法和拉格朗日原理，聚类中心为对应类别中各数据点的平均值，同时为了使得算法收敛，在迭代过程中，应使最终的聚类中心尽可能的不变。

　　具体算法细节这里不再详述，感兴趣的话可以在网上搜索，资料非常多。大家只要知道这个算法可以把数据聚为K类即可。这里的K需要在聚类开始前指定。

## 2.2.3 代码详解

```	matlab
clear all;
close all;
clc;

im1   = imread('./pic/bern_1.bmp');
im2   = imread('./pic/bern_2.bmp');
im_gt = imread('./pic/bern_gt.bmp');

im1   = double(im1(:,:,1));
im2   = double(im2(:,:,1));
im_gt = double(im_gt(:,:,1));
```

　　一般写matlab程序，上面都要加上 clear all;  close all; clc; 分别是清除内存中的变量，关闭打开的窗口，清除屏幕上的信息。接下来使用imread分别读入时相一、时相二和groud truth图片。因为图像是以BMP格式存储的，有3个通道，但SAR图像一般是纯灰度，这里使用 im1   = double(im1(:,:,1)); 来读取第一个通道的数据。

　　读入图像后，可以使用imshow函数查看输入的图像，比如说 imshow(im1, []); 因为这里已经把图像从uint8转化为了double类型，需要使用 [] 把图像最大值拉到255，最小值拉到0来进行显示。显示结果如下：

![](http://ww1.sinaimg.cn/large/6deb72a3ly1g1yighof15j20db0d9gpc.jpg)

<p align=center>图2.3 时相一图像显示结果</p>

```python
% 使用PCA的图像块大小, 必须为奇数
h = 3;
% feature vector, S <= h*h
S = 4; 

% 计算差分图像
im_di = abs(log10((im1+1)./(im2+1)));
```

　　接下来是算法两个重要的参数。进行变化检测时，需要考虑空间邻域的信息，需要在每一个像素周围取一个图像块，参数h表示该图像块的大小，即在差分图像上每个像素周围取一个 3x3 大小的图像块。S表示对该图像块利用PCA降维后，保留多少维特征。对于 3x3 大小的图像块，共有9个像素，因此特征为9维，算法会利用PCA将特征降维到4维。

　　对于输入的两个图像，使用log-ratio算子，计算得到差分图像（difference image），计算方法为：

　　$$ I_D = |log(I_1)-log(I_2)|$$

　　同时，还可以使用imshow(im_di, [])来查看差分图像，结果如下：

![](http://ww1.sinaimg.cn/large/6deb72a3ly1g1yiilpzumj20cv0d041g.jpg)

<p align=center>图2.4  差分图像显示结果</p>

　　接下来，在差分图像周围扩展像素（因为最外边的一圈像素，无法取到3x3的图像块），同时，把每个像素周围3x3的图像块中的像素值，保存到patterns中，代码为：

```matlab
% 在 im_di 周围扩展像素
im_di_new = zeros(ylen+h-1, xlen+h-1);
im_di_new( (h+1)/2:ylen+(h-1)/2, (h+1)/2:xlen+(h-1)/2) = im_di;

for j = 1:ylen
    for i = 1:xlen
        data = im_di_new(j:j+h-1, i:i+h-1);
        patterns( :, (j-1)*xlen+i) = data(:);      
    end
end
```

　　接着，使用PCA对提取到的邻域特征进行降维，并使用k-means方法将该特征聚为两类：

```python
[mean_vector, eigen_vectors] = PCA(patterns,S);
patterns = eigen_vectors * patterns;

fprintf('... ... k-means clustering ... ...\n');
idx = kmeans(patterns', 2, 'EmptyAction', 'singleton');
```

　　聚类以后，idx中就存储有相应的标签（1和2），代表聚类结果。但是，k-means方法只能够对数据进行聚类，并不能告诉我们结果中，哪一类是变化类，哪一类是非变化类。

　　所以，我们要计算这些像素在原差分图像中的均值，如果较大，是变化类；如果较小，是非变化类，代码如下：

```python
data1 = im_di(find(im_res==1));
data2 = im_di(find(im_res==2));
if mean(data1) < mean(data2)
    im_res = im_res - 1;
else
    im_res = ~(im_res - 1);
end
```

　　最后，把变化检测的结果保存并输出，同时，把准确率相关的信息保存到文件res.txt中，相关代码不再详述。

　　运行代码，得到变化检测结果为：

　　... ... 虚警像素: 158 

　　... ... 漏检像素: 145 

　　... ... 总体错误: 303 

　　... ... 准确率:   0.996656 

　　准确率为99.67%，是非常高的。还可以使用imshow(im_res, []); 来对变化检测的结果进行可视化，结果如下：

![](http://ww1.sinaimg.cn/large/6deb72a3ly1g1yijpl23cj20ct0d2aa4.jpg)

<p align=center>图2.5 变化检测结果</p>

　　评价变化检测的结果，一般可以使用FA，MD，OE，PCC来计算，其中FA指的是虚警，即不变类被错误判断成了变化类；MD指的是漏检，即变化类被错误的判断成了不变类；OE指的是FA和MD的和。相应的准确率，就是$(N-OE)/N \times100\%$ 。










