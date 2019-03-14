#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
@date 2019/3/12
"""
# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import re
# import time
# import nltk
# from nltk.corpus import stopwords
# from tensorflow.python.layers.core import Dense
# from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
# import tensorflow.contrib.seq2seq as seq

# nltk.download() # 下载nltk数据 path:E:\work\golaxy_job\golaxy_job\python_job\pycharm\dataset\nltk

# lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)
# # state 含有state.c state.h
# state = lstm.zero_state(batch_size=20,dtype=tf.float32)
#
# # 多层循环神经网络+层级dropout
# multirnn_output,multirnn_state = tf.nn.rnn_cell.MultiRNNCell(
#     [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(32),
#                                    input_keep_prob=0.8,output_keep_prob=0.7)
#      for i in range(10)]  # 10层
# )