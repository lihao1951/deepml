#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
线性回归
Author LiHao
Time 2018/10/10 9:42
"""
import os
import sys
import numpy as np
import tensorflow as tf

class Linear():
    def __init__(self):
        """
        初始化方法
        """
        self.W,self.b = self._get_variable()
        self.sess = tf.Session()
    def _get_variable(self):
        """
        获取变量
        :return:
        """
        W = tf.Variable(tf.zeros([2,1]),dtype=tf.float32,name="weights")
        b = tf.Variable(0.0,dtype=tf.float32,name="bias")
        return W,b
    def inference(self,X):
        """
        推断模型
        :param X:
        :return:
        """
        return tf.matmul(X,self.W) + self.b
    def loss(self,X,Y):
        """
        损失函数
        :param X:
        :param Y:
        :return:
        """
        Y_p = self.inference(X)
        loss_value = tf.reduce_sum(tf.squared_difference(Y,Y_p,name="squared_difference"))
        return loss_value
    def inputs(self,data=None):
        """
        得到初始输入数据-训练集
        :param data:
        :return:
        """
        if data is None:
            weight_age=[[66,46],[77,56],[46,36],[71,22],[46,16],[53,24],[73,43],[47,29],[36,6],[110,31]]
            blood_fat_content = [384,399,250,326,210,264,451,220,318,240]
        else:
            weight_age = data[0]
            blood_fat_content = data[1]
        return tf.to_float(weight_age),tf.to_float(blood_fat_content)

    def train(self,total_loss):
        """
        参数优化
        :param total_loss:
        :return:
        """
        learning_rate = 0.00001
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
        return optimizer
    def evaluate(self,X,Y):
        print(self.sess.run(self.inference([[80.0,25.0]])))
        print(self.sess.run(self.inference([[70.0,25.0]])))

    def result(self):
        saver = tf.train.Saver()
        X,Y = self.inputs()
        self.sess.run(tf.global_variables_initializer())
        loss_value = self.loss(X,Y)
        train_op = self.train(total_loss=loss_value)
        coord = tf.train.Coordinator()  # 线程管理器
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)  # 入队线程
        initial_steps = 0
        training_steps = 100
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            # 依据检查点文件的名称恢复全局迭代次数
            initial_steps = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        cur_loss = 0
        for step in range(initial_steps, training_steps):
            self.sess.run([train_op])
            if step % 1000 == 0 and step>1000:
                # 每迭代1000次就会保存最新的训练环境
                saver.save(self.sess, os.path.join(os.path.curdir+"\\my-model\\"), global_step=step)
            if step % 10 == 0:
                # 迭代10次就会输出当前的损失
                cur_loss = self.sess.run([loss_value])
                print("Loss:\t", cur_loss)
        self.evaluate(X,Y)
        coord.request_stop()
        coord.join(threads)
        saver.save(self.sess, os.path.join(os.path.curdir+"\\my-model\\"), global_step=training_steps)
        self.sess.close()

linear = Linear()
linear.result()