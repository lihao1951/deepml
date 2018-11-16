#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/10/15 10:41
"""
import os
import sys
import re
rules = {"data":[{"type":"aera","info":{"class01":"china","class02":"qinghai","class03":"xining"}}],
         "keyrule":["qinghai|xining&chengzhong|chengdong"]}
doc="zai xining de chengzhong"
print("doc is :\t",doc)
mingzhong = []
words = doc.split(" ")
set_words = set(words)
keyrule = rules.get("keyrule")
data = rules.get("data")
for rs in keyrule:
    ii = keyrule.index(rs)
    huos = rs.split("|")
    isAdd = False
    for yu in huos:
        yus = set(yu.split("&"))
        xx = yus.intersection(set(words))
        print(xx)
        if len(xx) == len(yus):
            isAdd=True
    if isAdd:
        mingzhong.append(data[ii])
print(mingzhong)


