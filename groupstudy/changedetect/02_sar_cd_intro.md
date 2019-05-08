## 第二章  遥感图像变化检测

　　遥感，指的是利用传感器或遥感器对地物的电磁波辐射、反射特性的一种非接触性、远距离探测技术。通过处理和分析遥感图像，人们可以获得其所表征的地物传达出的各种各样的信息。遥感图像包括光学遥感图像、合成孔径雷达(Synthetic aperture radar, SAR)图像、多光谱图像、高光谱图像等等，这些图像是遥感领域不可或缺的研究资源。
　　通过使用多时相遥感图像来检测发生在地球表面的变化是遥感技术最重要的应用之一。近年来，随着遥感技术的不断发展与成熟，人们可以通过多时相遥感图像的变化检测来对地表所发生的变化开展监测。变化检测主要通过对某一相同区域的不同时刻所对应的遥感图像进行分析，从而获知图像间的差异，进一步识别出地物状态信息的变化。这个课题对于农业的调查、城市发展研究、森林资源的监测以及救灾工作等方面具有重要而深远的意义。


## 2.1 问题描述

　　典型的变化检测如图2.1所示，左右两图分别拍摄于不同的时间。从图中可以看出，图中的部分树林被转化成了农业用地。![](http://ww1.sinaimg.cn/large/6deb72a3ly1g1y94j0xoaj20ok0dye7b.jpg)

<p align=center>图1.1 变化检测示例</p>

　　合成孔径雷达 ( Synthetic Aperture Radar, SAR ) 技术已经成为目前雷达技术领域先进的手段，渐渐成为该领域图像源的重要获取途径，它突破了传统雷达的功能模式，成为现代民生和军用科技强有力的武器。目前，SAR 图像技术逐渐发展成熟，相对与传统的光学遥感图像，SAR 系统具有不可比拟的特点，其分辨力不断提高，且兼具全天时、全天侯、大面积覆盖率等优势。基于此，SAR图像的应用范围日益宽广，常常被用来进行国土监测，城市规划变动，森林与农业监测，医疗检测，视频监控，军事行动等，尤其是自然灾害发生时，高效的变化检测技术能够及时地避免或减少人身和财务的损失。

　　同一地区的光学图像和SAR图像对比如图2.2所示，从图中可以看出，在光学图像中，受云遮挡，地物不可见。但SAR图像对云有较好的穿透性，地物清晰可见。

![](http://ww1.sinaimg.cn/large/6deb72a3ly1g1ya4rwommj20oc0c3qrg.jpg)

<p align=center>图2.2 光学图像与SAR图像的对比</p>

　　初期的变化检测算法依赖于先验知识，需要参考地物的特点来完成变化检测。这样就构建出许多**监督的方法**。但对于因突发灾害而需要的变化检测任务，地表发生变化后特征信息极为复杂且难以预先估计，而现场实地考察又会浪费大量的时间，造成救灾工作的严重滞后，这与变化检测技术的发展初衷背道而驰。因此，近年来人们更加重视不需要任何地物先验信息的无监督变化检测技术。

　　Bruzzone和Prieto于2002年提出，无监督遥感图像变化检测主要包括**三个步骤**：1）预处理；2）利用某算子产生反映图像变化情况的差异图；3）利用分类算法对差异图进行分析，得到最终的change map。

　　在差异图生成步骤中，现有的算法基本上都是基于**对数比值算子**（log-ratio operator），由于对数运算的压缩性，噪声将会被进一步抑制。

　　在差异图分类步骤中，应用最多的为**阈值法、聚类法**。**阈值法**容易操作，利用某个模型自动确定最优阈值。Bazi等于2005年使用Kittle-Illingworth（KI）准则自动确定阈值。Moser和Serpico在2006年提出了广泛KI准则法，可以自动确定阈值。**聚类法**一般使用Fuzzy c-means（FCM）算法或者 k-mans 算法，自动的把差异图聚为变化类和不变类两类。







