#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
数据流图 -利用tensorboard
Author LiHao
Time 2018/10/6 15:32
"""
import os
import sys
import numpy as np
import tensorflow as tf
#设置数据流图
g = tf.get_default_graph()#默认
#g = tf.Graph()#重新设置新的数据流图
with g.as_default():
    #在相应的数据流图下设置数据流
    #a = tf.constant(5,name="input_a")#常量
    a = tf.placeholder(tf.int32,shape=[2],name="my_input")#占位符
    """
    常量和占位符都是不可变的，可变的对象为Variable Op
    可定义tf.Variable() tf.zeros() tf.ones() tf.random_nomal() tf.random_uniform()等Op 都接受一个shape参数
    由于Variable是由Sess管理的，所以必须在一个Session对象中进行初始化:
    init = tf.global_variables_initializer()---新版本 / tf.initialize_all_variables()---旧版本
    sess = tf.Session()
    sess.run(init)
    
    若要修改Variable的值，需要手动分配
    a=tf.Variable(5,name="input_a")
    a.assign(10) 
    自增 assign_add 自减 assign_sub
    利用sess.run(a.assign(10))来实现更改
    在tf的机器学习模型中，Variable的更改已自动完成，无需手动
    若要设置该变量只可以手动修改，无法自动 则需要设置trainable=False
    a=tf.Variable(5,trainable=False,name="input_a")
    
    可定义name scope来规范显示各个阶段的流程
    with tf.name_scope("A"):
        ...
    """

    b = tf.constant(3,name="input_b")
    c = tf.reduce_sum(a,name="sum_c")
    d = tf.add(c,b,name="add_d")
    sess = tf.Session(graph=g)#默认空是默认的数据流，还可以指定特定的数据流
    output= sess.run(d,feed_dict={a:np.array([5,3],dtype=np.int32)})
    train_writer = tf.summary.FileWriter('./train', g)
    sess.close()#释放资源