#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
"""
import os
import sys
import math

def guiyi(T,X,P=200):
    return P*math.atan(T*math.sqrt(X))/math.pi

def chuanbo(T,comment_num,passport_num):
    X = 1*comment_num +2*passport_num
    return guiyi(T,X)

def xianzhu(T,web_level,web_board,title_site,pt_level,one_num,two_num,three_num):
    X=web_level*web_board*title_site*pt_level*(3*one_num+2*two_num+1*three_num+10)
    return guiyi(T,X)

def fengfu(T,con_len,is_pic,isvedio,ismuic):
    c = guiyi(0.05,con_len,P=2)
    X=math.pow(2,(3*c*(1*is_pic+1*ismuic+1.5*isvedio+1)))
    return guiyi(T,X)

def haoping(T,sent,bi):
    X=sent*3*3*(bi+0.01)
    return guiyi(T,X)

print("chuanbo ",chuanbo(0.2,200,200))
print("quanwei ",xianzhu(0.15,3,1,1,2,1,1,1))
print("haoping ",haoping(0.2,90,0.6))
print("fengfu ",fengfu(0.35,1000,1,0,0))