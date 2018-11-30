#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/11/26 11:12
"""
import os
import sys
import numpy as np


def mydistance(A, B,isEculidean=True):
    """
    计算数据之间的距离
    :param A:
    :param B:
    :return:
    """
    if isEculidean:
        if len(B.shape) == 1:
            return np.sqrt(np.sum(np.power(B - A, 2), axis=0))
        return np.sqrt(np.sum(np.power(B - A, 2), axis=1))
    else:
        if len(B.shape) == 1:
            M = np.sum(A * B)
            BN = np.linalg.norm(B)
        else:
            M = np.sum(A * B,axis=1)
            BN = np.linalg.norm(B, axis=1)
        AN = np.linalg.norm(A)
        Z = AN*BN
        result = 1- M/Z
        return result