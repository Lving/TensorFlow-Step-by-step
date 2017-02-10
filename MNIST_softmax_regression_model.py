# __author__ = "Administrator"
# -*- coding: utf-8 -*-
# @description: implement a softmax regression model upon MNIST handwritten digits
# @ref: http://yann.lecun.com/exdb/mnist/

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data", one_hot=False)
train_x_minmax = mnist.train.images
train_y_data = mnist.train.labels
test_x_minmax = mnist.test.images
test_y_data = mnist.test.labels


# we evaluate the softmax regression model by sklearn first
eval_sklearn = True
if eval_sklearn:
    print("Start evaluating softmax regression model by sklearn...")
    reg = LogisticRegression(solver="lbfgs", multi_class="multinomial")
    reg.fit(train_x_minmax, train_y_data)
    np.savetxt('coef_softmax_sklearn.txt', reg.coef_, fmt='%.6f')
    test_y_predict = reg.predict(test_x_minmax)
    print('Accuracy of test set: %f' % accuracy_score(test_y_data, test_y_predict))

eval_tensorflow = True
batch_gradient = False
if eval_tensorflow:
    print("Start evaluating softmax regression model by tensorflow...")
    # reformat y into one-hot encoding style
    lb = preprocessing.LabelBinarizer()
    lb.fit(train_y_data)
    train_y_data_trans = lb.transform(train_y_data)
    test_y_data_trans = lb.transform(test_y_data)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    V = tf.matmul(x, W) + b
    y = tf.nn.softmax(V)

    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    if batch_gradient:
        for step in range(300):
            sess.run(train, feed_dict={x: train_x_minmax, y_: train_y_data_trans})
            if step % 10 == 0:
                print("Batch Gradient Descent processing step %d" % step)
            print("Finally we got the estimated results, take such a long time...")
    else:
        for step in range(1000):
            sample_index = np.random.choice(train_x_minmax.shape[0], 100)
            batch_xs = train_x_minmax[sample_index, :]
            batch_ys = train_y_data_trans[sample_index, :]
            sess.run(train, feed_dict={x: batch_xs, y_:batch_ys})
            if step % 100 == 0:
                print("Stochastic Gradient Descent processing step %d" % step)

    np.savetxt('coef_softmax_tf.txt', np.transpose(sess.run(W)), fmt='%.6f')
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy of test set: %f" % sess.run(accuracy, feed_dict={x: test_x_minmax, y_:test_y_data_trans}))

