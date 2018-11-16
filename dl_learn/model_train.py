#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/8/28 14:50
"""
import os
import sys
import tensorflow as tf

def linear_train():
    """
    训练线性回归模型
    math_base tf
    :return:
    """
    #定义变量 该变量是最终需要的参数
    W = tf.Variable([-0.2],dtype=tf.float32)
    b = tf.Variable([0.7],dtype=tf.float32)
    #定义占位符 该占位符为需要输入的参数 即：训练集
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(dtype=tf.float32)
    # 设定训练集
    x_train = [1, 2, 3, 5]
    y_train = [0, -1, -2, -4]
    #定义模型
    linear_model = W * x + b
    #定义优化器，这里选用梯度下降优化 学习速率为 learning_rate 指定
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    #定义损失函数 这里设定为均方误差
    loss = tf.reduce_sum(tf.square(linear_model - y))
    #定义需要有优化的函数
    train = optimizer.minimize(loss)
    #设置会话函数 初始化全局变量
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    #迭代优化
    for i in range(1000):
        sess.run(train,{x:x_train,y:y_train})
    # 输出当前的结果
    curr_W,curr_b,curr_loss = sess.run([W,b,loss],{x:x_train,y:y_train})
    print("W:%s , b:%s ,loss:%s " % (curr_W,curr_b,curr_loss))

def mnist_data():
    from tensorflow.examples.tutorials.mnist import input_data
    # 读取MNIST数据集
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    return mnist

def mnist_train():
    """
    训练mnist分类器
    math_base tf softmax 单层网络
    :return:
    """
    mnist = mnist_data()
    #定义占位符
    x = tf.placeholder(tf.float32,shape=[None,784])
    y_ = tf.placeholder(tf.float32,shape=[None,10])
    #定义权重及偏置
    W = tf.Variable(tf.zeros([784,10]),dtype=tf.float32)
    b = tf.Variable(tf.zeros([10]),dtype=tf.float32)
    #定义训练网络及输出值
    y = tf.nn.softmax(tf.matmul(x,W)+b)
    #定义交叉熵
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    #定义优化器
    optimizer = tf.train.GradientDescentOptimizer(0.2)
    #定义训练步骤
    train = optimizer.minimize(cross_entropy)
    #定义会话 并初始化 变量
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    #按批次进行训练
    for _ in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100)
        sess.run(train,feed_dict={x:batch_x,y_:batch_y})
    #评估模型
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Test Accuracy: ",sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
    print(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

def weight_variable(shape):
    """
    初始化权重变量
    :param shape:
    :return:
    """
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    初始化偏置变量
    :param shape:
    :return:
    """
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    """
    设置二维卷积
    设置SAME填充边界->得到的输入和输出尺寸一样
    :param x:
    :param W:
    :return:
    """
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    """
    设置最大池化层
    设置2*2 -> 1*1 大小的池化 变为原来面积的1/4
    :param x:
    :return:
    """
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def cnn_mnist_train():
    """
    简单的卷积神经网络
    4层 -->  2层卷积 2层全连接
    :return:
    """
    mnist = mnist_data()
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.float32,[None,10])
    # shape=[数量,行数,列数,通道数]
    x_image = tf.reshape(x,shape=[-1,28,28,1])
    #第一个卷积层
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    # 先进行卷积
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

    #dropout

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(600):
        batch = mnist.train.next_batch(100)
        if i%100 == 0:
            train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
            print("step %d , train accuracy %s" % (i,train_accuracy))
        sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.3})
    print("Test accuracy %s" % sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:0.5}))

def autoencoder_train():
    pass

if __name__ == "__main__":
    # linear_train()
    # mnist_train()
    # import platform
    # print(platform.system())
    # print(os.path.dirname(__file__))
    cnn_mnist_train()