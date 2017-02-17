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

#### TensorBoard之Embedding
TensorBoard是TensorFlow自带的可视化工具， Embedding是其中的一个功能， 用于在二维或者三维空间对高维数据进行探索。  
> An embeding is a map from input data ti points in euclidean space  

首先，从[这里](https://www.tensorflow.org/images/mnist_10k_sprite.png)下载图片，放到主目录的log目录下； 然后执行代码； 最后， 执行下面的命令启动TensorBoard.
```
tensorboard --logdir=log
```
执行后， 在浏览器中输入，命令行中出现的链接地址。  
```
Starting TensorBoard 39 on port 6006
(You can navigate to http://....)
```  
[这篇文章](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)对MNIST的可视化做了深入的研究， 非常值得细读。

### 一些tensorflow的基本操作
详细参考这篇[文章](http://blog.csdn.net/lenbow/article/details/52152766)

1. tensorflow的基本运作
```
import tensorflow as tf
 #定义‘符号’变量，也称为占位符
a = tf.placeholder("float")
b = tf.placeholder("float")
y = tf.mul(a, b) #构造一个op节点

sess = tf.Session()#建立会话
#运行会话，输入数据，并计算节点，同时打印结果
print sess.run(y, feed_dict={a: 3, b: 3})
# 任务完成, 关闭会话.
sess.close()
```
2. tf函数


操作组 | 操作
---|---
Maths | Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal
Array | Concat, Slice, Spilt, Constant, Rank, Shape, Shuffle
Matrix | MatMul, MatricInverse, MatrixDeterminant
Neuronal Network | SoftMax, Sigmoid, ReLU, Convonlution2D, MaxPool
Checkpoint | Save, Restore
Queues and syncronizations | Enqueue, Dequeue, MutexAcquire, MutexRelease
Flow control | Merge, Switch, Enter, Leave, NextIteration

TensorFlow的算术操作如下：
操作 | 描述
---|---
tf.add(x, y, name=None) | 求和
tf.sub(x, y, name=None) | 减法
tf.mul(x, y, name=None) | 乘法
tf.div(x, y, name=None) | 除法
tf.mod(x, y, name=None) | 取模
tf.abs(x, name=None) | 绝对值
tf.neg(x, name=None) | 取负
tf.sign(x, name=None) | 符号函数 
tf.inv(x, name=None) | 取反
tf.square(x, name=None) | 平方
tf.round(x, name=None) | 舍入最接近的整数
tf.pow(x, y, name=None) | 幂次方

张量操作 Tensor Transformations
* 数据类型转换Casting

操作 | 描述
--- | ---
tf.string_to_number(string_tensor, out_type=None, name=None) | 字符串转为数字
tf.to_double(x, name='ToDouble') | 转为64位浮点类型
tf.to_float(x, name=ToFloat) | 转为32位浮点类型
tf.to_int32(x, name='Toint32') | 转为32位整型
tf.cast(x, dtype, name=None) | 将x或者x.value转换成dtype (tensor `a` = [1.8, 2.2] tf.cast(a, tf.int32))


* 形状操作shape  

操作 | 描述
---|---
tf.shape(input, name=None) | 返回数据的shape
tf.size(input, name=None) | 返回数据的元素数量
tf.rank(input, name=None) | 返回tensor的rank, 此rank不同于矩阵的rank, t=[[[1,1,1], [2,2,2]],[[3, 3, 3], [4, 4, 4]]] , shape of t is [2,2,3], rank(t) =3
tf.reshape(tensor, shape, name=None) | 改变tensor的形状, 如果shape有元素[-1], 表示在该维度打平至一维 
tf.expand_dims(input, dim, name=None) | 插入维度1进入tensor中  

* 矩阵操作  

操作 | 描述
---|---
tf.diag(diagonal, name=None) |
tf.diag_part(input, name=None) |
tf.trace(x, name=None) | 求一个2维tensor的迹， 即对角值diagonal之和
tf.transpose() | 转置
tf.matmul() | 矩阵相乘
tf.matrix_determinant() | 返回矩阵的行列式
tf.matrix_inverse() | 求矩阵的逆矩阵
tf.matrx_solve() | 对矩阵求解  

* 归约计算(reduction)  

操作 | 描述
---|---
tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None) | 计算输入tensor元素的和， 或者按照reduction_indices指定的轴进行求和
tf.reduce_prod(input_tensorm reduction_indices=None, keep_dims=False, name=None) | 计算输入tensor元素的乘积， 或者按照reduction_indices指定的轴进行求乘积
tf.reduce_min() | 求tensor中的最小值
tf.reduce_max() | 求tensor中的最大值
tf.mean() | 求tensor的平均值
tf.reduce_all() | 对tensor中的各个元素进行求逻辑`and`
tf.reduce_any() | 对tensor中的各个元素求逻辑`or`  
tf.accumulate_n(inputs, shape=None, tensor_dtype=None, name=None) | 计算一系列tensor的和
tf.cumsum() | 求累积和

* 分割(Segmentation)
* 序列比较与索引提取  

操作 | 描述
---| ---
tf.argmin(input, dimension, name=None) | 返回input最小值的索引index
tf.argmax() | 返回最大值索引

神经网络(NN)
* 激活函数(Activation Function)  

操作 | 描述
---|---
tf.nn.relu(features, name=None) | 整流函数： max(features, 0)
tf.nn.relu6(features, name-None) | 以6为阈值的整理函数：min(max(features, 0), 6)
tf.nn.elu(features, name=None) | elu函数, exp(features) - 1 if < 0, 否则features Exponential Linear Units
tf.nn.softplus(features, name=None) | 计算softplus:log(exp(features) + 1)
tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None) | 计算dropout, keep_prob为keep概率
tf.nn.bias_add(value, bias, data_format=None, name=None) | 对value加一偏置量， 此函数为tf.add的特殊情况，bias仅为一维，函数通过广播机制进行与value求和
tf.sifmoid(x, name=None) | y = 1 / (1 + exp(-x))
tf.tanh(x, name=None) | 双曲线切线激活函数  

* 卷积函数(convolution)  

操作 | 描述
---|---
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None) | 在给定4D input 与 filter 下计算2D卷积， 输入shape为[batch, height, weight, in_channels]
tf.nn.conv3d(input, filter, strides, padding, name=None) | 在给定5D input 与filter下计算3D卷积，输入shape为[batch, in_depth, in_height, in_width, in_channels]  

* 池化函数(pooling)

操作 | 描述
---|---
tf.nn.avg_pool(value, ksize, strides, padding, date_format, name=None) | 平均方式池化
tf.nn.max_pool() | 最大值方式池化
tf.nn.max_pool_with_argmax(input, ksize, padding, Targmax, name) | 返回一个二维元祖(output, argmax), 最大值pooling， 返回最大值及其相应索引
tf.nn.avg_pool3d(input, ksize, strides, padding, name) | 3D平均值pooling
tf.nn.max_pool3d() | 3D最大值pooling  

* 数据标准化  

操作 | 描述
---|---
tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None) | 对维度dim进行L2范式标准化， output = x / sqrt(max(sum(x**2), epsilon))
tf.nn.sufficient_statistics(x, axes, shift=None, keep_dims=False, name=None)| 计算与均值和方差相关的完全统计量， 返回4维元祖（元素个数， 元素总和， 元素的平方和， shift结果）
tf.nn.normalize_moments(counts, mean_ss, variance_ss, shift, name=None) | 基于完全统计量计算均值和方差
tf.nn.moments(x, axes, shift=None, name=None, keep_dims=False) | 直接计算均值与方差  

* 损失函数  

操作 | 描述
---|---
tf.nn.l2_loss | output = sum(t**2) / 2  

* 分类函数  

操作 | 描述
---|---
tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None) | 计算输入logits, targets的交叉熵
tf.nn.softmax(logits, name=None) | 计算softmax softmax[i,j] = exp(logits[i, j]) / sum_j(exp(logits[i,j]))
tf.nn.log_softmax(logits, name=None) | logsoftmax[i,j] = logits[i,j] - log(sum(exp(logits[i])))
tf.nn.softmax_cross_entropy_with_logits(logits, label) | 计算logits和label的softmax的交叉熵， logits， label必须为相同的shape与数据类型
tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label, name=None) | 计算logits和label的softmax交叉熵
tf.nn.weighted_cross_entropy_with_logits(logits, targets, pos_weight, name=None) | 与sigmoid_cross_entropy_with_logits()相似， 但给正向向本损失加了权重pos_weight  






