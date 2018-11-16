#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/11/4 12:28
"""
import os
import sys
import matplotlib.pyplot as plt
import random
from math import fabs
def diff(X=[]):
    o = X[0]
    r = []
    for i in X:
        r.append(i-o)
    return r

def seq(X=[]):
    sum = 0
    for i in range(len(X)-1):
        sum += X[i]
    sum = sum + 0.5*X[-1]
    return sum

def simi(S0,Si):
    T = 1.0+ abs(S0) + abs(Si)
    P = abs(S0-Si)
    return T/(T+P)

def plotimage(X,*gras):
    for x in gras:
        plt.plot(X,x)
    plt.show()

def ptest():
    a=[1,11,1,31,71,101,161,89]

    b = []
    for aa in a:
        b.append(aa*2.5 + random.randint(-15,55))
    c = [52,67,25,1112,87,898,364,9]
    d = [1000,900,800,700,600,500,400,320]
    s0=seq(diff(a))
    si=seq(diff(b))
    sc = seq(diff(c))
    sd = seq(diff(d))
    print(simi(s0,si))
    print(simi(s0,sc))
    print(simi(sc,sd))
    plotimage([1,2,3,4,5,6,7,8],a,b,c,d)

import math
print(-(1/10.0)*math.log2(1/10.0)*10)