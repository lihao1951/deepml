#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
可以编译文件为so文件
供其余程序调用
Author LiHao
Time 2018/8/20 15:41
"""
import os
import sys
from Cython.Build import cythonize
from distutils.core import setup
setup(ext_modules = cythonize("test_so.py"))