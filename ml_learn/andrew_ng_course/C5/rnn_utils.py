#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/11/5 9:23
"""
import os
import numpy as np
def softmax(x):
    """
    求各个数据的softmax比率
    :param x:
    :return:
    """
    e_x = np.exp(x-np.max(x))#减去最大值，得到的值就为属于（0,1）之间的值
    return e_x / np.sum(e_x,axis=0) #按行去加

def sigmoid(x):
    """
    激活函数
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))

def initialize_adam(parameters):
    """
    初始化变化量的值
    dW db
    :param parameters:
    :return:
    """
    L = len(parameters)
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
    return v,s
