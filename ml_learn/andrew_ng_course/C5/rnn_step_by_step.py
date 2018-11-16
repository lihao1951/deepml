#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/11/5 9:19
"""
import os
import sys
import numpy as np
from ml_learn.andrew_ng_course.C5.rnn_utils import *

def rnn_cell_forward(xt,a_prev,parameters):
    """
    单个神经元RNN正向传播
    输入前方的x[t]和a[t-1]及参数Wax,Waa,Wya,ba,by
    :param xt:
    :param a_prev:
    :param parameters:
    :return:
    """
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    a_next = np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,xt)+ba)
    yt_pred = softmax(np.dot(Wya,a_next)+by)
    cache = (a_next,a_prev,xt,parameters)
    return a_next,yt_pred,cache

def rnn_forward(x,a0,parameters):
    """
    :param x:shape->(nx,,Tx)
    :param a0:(na,m)
    :param parameters: Wax,Waa,Wya,ba,by
    :return:
    """
    caches = []
    n_x,m,T_x = x.shape
    n_y,n_a = parameters["Wya"].shape
    a = np.zeros([n_a,m,T_x])
    y_pred = np.zeros([n_y,m,T_x])
    a_next = a0

    for t in range(T_x):
        a_next,yt_pred,cache = rnn_cell_forward(xt=x[:,:,t],a_prev=a_next,parameters=parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)

    caches = (caches,x)
    return a,y_pred,caches

def lstm_cell_forward(xt,a_prev,c_prev,parameters):
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    n_x,m = xt.shape
    n_y,n_a = Wy.shape
    concat = np.zeros([n_a+n_x,m])
    concat[:n_a,:] = a_prev
    concat[n_a:,:] = xt
