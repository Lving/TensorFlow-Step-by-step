# __author__ = "Administrator"
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing

x_data = np.loadtxt('D:\Learning_code\TensorFlow_\data\ex3x.dat').astype(np.float32)
y_data = np.loadtxt('D:\Learning_code\TensorFlow_\data\ex3y.dat').astype(np.float32)
# x_data.shape (47,2)
# y_data.shape(47,)

# we evalute the x and y by sklearn to get a sense of thr coefficients
reg = linear_model.LinearRegression()
reg.fit(x_data, y_data)
print("Coefficients of sklearn : k=%s, b=%f" % (reg.coef_, reg.intercept_))

# Now we use tensorflow to get similar results

# before we put the x_data into tensorflow, we need to standardize it
# in order to archieve better performance in gradient descent
# if not standardized, the convergency speed could not be tolerated
# Reasons: if a feature has a variance that is orders of magnitude larger than others
# it might dominate the objective function
# and make the estimator unable to learn from other features correctly as expected

scaler = preprocessing.StandardScaler().fit(x_data)
x_data_standard = scaler.transform(x_data)

W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1, 1]))
y = tf.matmul(x_data_standard, W) + b

loss = tf.reduce_mean(tf.square(y - y_data.reshape(-1, 1))) / 2
optimizer = tf.train.GradientDescentOptimizer(0.3)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(1000):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(W).flatten(), sess.run(b).flatten())

print("Coefficients of tensorflow (input should be standardized): K=%s, b=%s" % (sess.run(W).flatten(), sess.run(b).flatten()))
print("Coefficients of tensorflow (raw input): K=%s, b=%s" % (sess.run(W).flatten() / scaler.scale_, sess.run(b).flatten() - np.dot(scaler.mean_ / scaler.scale_, sess.run(W))))
# 对于梯度下降算法， 变量是否标准化很重要。在这个例子中， 变量是一个面积，一个是房间数，两级相差很大，如果不归一化
# 面积在目标函数和梯度中就会占据主导地位，导致收敛极慢

