# __author__ = "Administrator"
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn import linear_model

x_data = np.loadtxt('D:\Learning_code\TensorFlow_\data\ex2x.dat')
y_data = np.loadtxt('D:\Learning_code\TensorFlow_\data\ex2y.dat')

# We use scikit-learn first to get a sense of coefficient
reg = linear_model.LinearRegression()
reg.fit(x_data.reshape(-1, 1), y_data) # x_data转换成列

print('Coefficient of scikit-learn linear regression: k=%f, b=%f' % (reg.coef_, reg.intercept_))

# Then we apply tensorflow to achive similar results
# the structure of tensorflow code can be divided into two parts

# first part: set up computation graph
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data)) / 2
optimizer = tf.train.GradientDescentOptimizer(0.07)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# second part: launch the graph
sess = tf.Session()
sess.run(init)

for step in range(1500):
    sess.run(train)
    if step % 100 == 0:
        print(step, sess.run(W), sess.run(b))
print("Coefficient of tensorflow linear regression: k=%f, b=%f" % (sess.run(W), sess.run(b)))