#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
MCMC 测试
@author LiHao
@date 2019/1/16
"""
import numpy as np
matrix = np.matrix([[0.9,0.075,0.025],[0.15,0.8,0.05],[0.25,0.25,0.5]],dtype=np.float32)
vector = np.matrix([[0.6,0.15,0.25]],dtype=np.float32)
for i in range(10):
    matrix = matrix.dot(matrix)
    print(matrix)