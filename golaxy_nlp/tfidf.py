#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/8/6 10:00
"""
import os
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
def getPackagePath():
    path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(path))
sys.path.append(getPackagePath())
from golaxy_nlp.dataload import validation_data

sentences = validation_data(True)
print("-*- 训练样本数：",len(sentences))
tfidf = TfidfVectorizer(max_features=500000)
model = tfidf.fit_transform(sentences)

with open("./model/tfidf","wb") as w:
    pickle.dump(tfidf,file=w)