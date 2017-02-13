# TensorFlow-Step-by-step
a comparison of sklean and tensorflow
#### 线性回归： 
对于tensorflow，梯度下降的步长alpha参数需要很仔细的设置，步子太大容易扯到蛋导致无法收敛；步子太小容易等得蛋疼。迭代次数也需要细致的尝试。 
#### 多元线性回归： 
对于梯度下降算法，变量是否表转化很重要。在这个例子中，变量一个是面积，一个是房间数，量级相差很大，如果不归一化，面积在目标函数和梯度中就会占据主导地位，导致收敛极慢
#### 逻辑回归：
* 对于逻辑回归，损失函数比线性回归模型复杂了一些。首先需要通过sigmoid函数，将线性回归的结果转化为0至1之间的概率。然后写出每个样本的发生概率（似然）， 那么所有样本的发生概率就是每个样本发生概率的乘积。为了求导方便，我们对所有样本的发生概率取对数，保持其单调性的同时，可以将连乘变成求和，对数极大似然估计方法的目标函数就是最大化所有样本的发生概率；机器学习的习惯将目标函数称为损失， 所以将损失定义为对数似然的相反数， 以转化为极小值问题。 

* 我们提到逻辑回归时，一般是指而分类问题；然后这套思想的是可以轻松就拓展为多分类问题的，在机器学习领域一般称之为softmax回归模型。

#### 基于MNIST数据的softmax regression 
* sklearn的估计时间有点长， 因为每一轮参数更新都是基于全量的训练集数据算出损失，再算出梯度，然后再改进结果的。  
* tensorflow采用batch Gradient Descent估计算法时，时间也比较长，原因同上。  
* tensorflow采用stochastic Gradient Descent估计算法时间短，最后的估计结果也挺好，相当于每轮迭代只用到了部分数据集算出损失和梯度，速度变快，但可能bias增加；所以把迭代次数增多，这样可以降低variance， 总体上误差相比Batch Gradient Descent 并没有差多少。

#### 基于MNIST数据的卷积神经网络CNN
* 参数数量：第一卷积层5*5*1*32=800个参数， 第二卷积层5*5*32*64=51200个参数，第三个全连接层7x7x64x1024=3211264个参数，第四个输出层1024x10=10240个参数，总量级为330万个参数，单机训练时间约为30分钟。
* 关于优化算法： 随机梯度下降算法的learning Rate需要逐渐减小，因为随机抽取样本引入了噪音，使得我们的最小点的随机梯度仍然不为0、对于batch gradient descent 不存在这个问题。在最小点损失函数的梯度变为0，因此BGD可以使用固定的learning rate。 为了使learning rate逐渐变小， 有一下几种变种算法：
`Momentum` algorithm accumulates an exponentially decaying moving average of past gradients and continues to move in their direction.  
`AdaGrad` adapts the learning rates of all model parameters by scaling them inversely proportional to the square root of the sum of all their historical squared values. But the accumulation of squared gradients from the beginning of training can result in a premature and excessive decrease in the effective learning rate.  
`RMSProp`： AdaGrad is designed to converge rapidly when applied to a convex function. When applied to a non-convex function to train a neural network, the learning trajectory may pass through many different structures and eventually arrive at a region that is a locally convex bowl. AdaGrad shrinks the learning rate according to the entire history of the squared gradient and may have made the learning rate too small before arriving at such a convex structure.
RMSProp uses an exponentially decaying average to discard history from the extreme past so that it can converge rapidly after finding a convex bowl, as if it were an instance of the AdaGrad algorithm initialized within that bowl.   
`Adam`：The name “Adam” derives from the phrase “adaptive moments.” It is a variant on the combination of RMSProp and momentum with a few important distinctions. Adam is generally regarded as being fairly robust to the choice of hyperparameters, though the learning rate sometimes needs to be changed from the suggested default.   
* 如果将MNIST数据集的AdamOptimizer换成GradientDescentOptimizer， 测试集的准确率为0.9296




