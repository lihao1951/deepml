#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/10/14 11:32
"""
import os
import sys
import numpy as np

a3=np.array([[0.2,0.4,0.5,0.6]],dtype=np.float32)
print("a3 first \n",a3)
d3=np.random.rand(a3.shape[0],a3.shape[1])<0.6
print(d3)
a3 = np.multiply(a3,d3)
print(a3)
a3 /= 0.6
print(a3)