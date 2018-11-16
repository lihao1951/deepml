#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/10/11 15:44
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def gradient_curve():
    X = np.array([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6],dtype=np.int8)
    y = np.power(X,2)+2*X+1
    plt.plot(X,y)
    plt.show()
def gradient_3d():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-1.2, 1.2, 0.1)
    Y = np.arange(-1.2, 1.2, 0.1)
    X, Y = np.meshgrid(X, Y)  # x-y 平面的网格
    Z = np.power(X,2)+np.power(Y,2)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
    #ax.contourf(X, Y, Z, zdir='z', offset=-2)
    ax.set_zlim(-2, 2)
    plt.show()

gradient_3d()