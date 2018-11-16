#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
工具实验模块
@author LiHao
"""
import os
import sys
def getPackagePath():
    path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(path))
sys.path.append(getPackagePath())
import deepnlp

deepnlp.hi()
from flask import Flask