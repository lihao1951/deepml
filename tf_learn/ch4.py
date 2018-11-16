#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
机器学习实现 基于 TensorFlow
Author LiHao
Time 2018/10/9 14:07
"""
import os
import sys
import tensorflow as tf
import numpy as np

"""
监督式学习可以用以下闭环的训练过程表示
（1）初始化模型参数及变量
（2）读取训练数据
（3）在训练数据上拟合模型，推断参数
（4）计算与期望输出的损失
（5）调整模型参数，并不断重复（3）-（5），直至结果满足要求、
可用如下通用代码框架描述
"""
def inference(X):
    """
    推断模型
    :param X:
    :return:
    """
    pass
def loss(X,Y):
    """
    计算损失
    :param X:
    :param Y:
    :return:
    """
    pass
def inputs():
    """
    读取或生成训练数据及期望输出
    :return:
    """
    pass
def train(total_loss):
    """
    依据计算的总损失训练或调整训练参数
    :param total_loss:
    :return:
    """
    pass
def evaluate(sess,X,Y):
    """
    对训练得到的模型进行评估
    :param sess:
    :param X:
    :param Y:
    :return:
    """
    pass
"""
可以借助tf.train.Saver()函数保存数据流图中的变量至专门的二进制文件中
应当周期性的检查所有的变量，创建检查点（checkpoint）文件，并从最近的检查点中恢复训练
"""
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    X,Y = inputs()
    total_loss = loss(X,Y)
    train_op = train(total_loss)
    coord = tf.train.Coordinator()#线程管理器
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)#入队线程
    initial_steps = 0
    training_steps = 1000
    #检查是否保存了检查点文件
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    #若含有检查点文件，则从最近的文件中回复
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        #依据检查点文件的名称恢复全局迭代次数
        initial_steps = int(ckpt.model_checkpoint_path.rsplit('-',1)[1])
    for step in range(initial_steps,training_steps):
        sess.run([train_op])
        if step % 1000 == 0 :
            #每迭代1000次就会保存最新的训练环境
            saver.save(sess,"my-model",global_step=step)
        if step % 10 == 0:
            #迭代10次就会输出当前的损失
            print("Loss:\t",sess.run([total_loss]))
    evaluate(sess,X,Y)
    coord.request_stop()
    coord.join(threads)
    #保存最后的训练环境 默认情况下只保存最近5个文件
    saver.save(sess,"my-model",global_step=training_steps)
    sess.close()