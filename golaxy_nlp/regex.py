#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/8/3 13:44
"""
import os
import sys
import re

def is_num(word):
    """
    正则匹配数字
    :param word:
    :return:
    """
    r = re.search(r'(\d+)',word)
    if r is not None:
        return True
    else:
        return False

def remove_illegal_mark(content):
    """
    去除特殊标记
    :param content:
    :return:
    """
    pattern = re.compile(r'<[^>]+>|{.*?}|【.*?】|www.*?[conmf]{2,3}|http[s]{0,1}', re.S)
    result = pattern.sub('', content.strip().replace("\n",""))
    return result
