#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/11/26 11:12
"""
import os
import sys
import numpy as np


def eculide(A, B):
    """
    计算数据之间的距离
    :param A:
    :param B:
    :return:
    """
    if len(B.shape) == 1:
        return np.sqrt(np.sum(np.power(B - A, 2), axis=0))
    return np.sqrt(np.sum(np.power(B - A, 2), axis=1))