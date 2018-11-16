#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/10/31 10:46
"""
import os
import sys
import platform
import tensorflow as tf

def __getCurrentPathAndOS__():
    """
    获取当前文件的路径及操作系统
    :return:
    """
    filename = __file__
    current_path = os.path.dirname(filename)
    os_name = platform.system()
    if os_name.lower().__contains__("win"):
        return "windows",current_path
    else:
        return "linux/mac",current_path
def load_mnist():
    MNIST_PATH = "MNIST_DATA"
    os_name,current_path = __getCurrentPathAndOS__()
    if os_name is "windows":
        current_path += '\\' + MNIST_PATH
    else:
        current_path += '/' +MNIST_PATH
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(current_path, one_hot=True)
    return mnist