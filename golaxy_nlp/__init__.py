#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
golaxy_nlp
@author LiHao
"""
import os
import sys

def deal_single_str():
    f = open('./data/comments.txt','r',encoding='utf-8')
    lines = f.readlines()
    count = 0
    w = open('./data/c.txt','a',encoding='utf-8')
    for line in lines:
        count += 1
        if count > 10000:break
        data = line.strip('\n').split(" ")
        p = []

        for d in data:
            for word in d:
                p.append(word)
        l = ' '.join(p)+'\n'
        print(l)
        w.write(l)
    w.close()
    f.close()

string_types = str