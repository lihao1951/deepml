#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/11/27 14:12
"""
import os
import sys
import jieba
from pyspark.mllib.feature import HashingTF,IDF,Word2Vec
from pyspark.context import SparkContext

def tfidf_train():
    sc = SparkContext(master="local[2]",appName="tfidfApp")
    words = sc.textFile("words.txt")
    documents = words.map(lambda x:x.split(" "))
    hashTf = HashingTF()
    tf = hashTf.transform(documents) # 转化为词频
    tf.cache()
    documents.foreach(print)
    idf = IDF()
    idf = idf.fit(tf) # 转化为idf
    tfidf = idf.transform(tf) # 转化为tfidf向量表示
    # a=（(1048576,[96460,113733,198291,377545,943550],[1.6094379124341003,1.6094379124341003,1.6094379124341003,1.6094379124341003,1.6094379124341003])）
    # a[0]:总词汇的大小,a[1]:每行文本每个词的索引,a[2]:每个索引词的tfidf权重
    tf.foreach(print)
    tfidf.foreach(print)
    sc.stop()

def word2vec_train():
    sc = SparkContext(master="local[2]", appName="w2vApp")
    file = sc.textFile("words.txt")#在linux上需要指定file:// 或者hdfs://
    documents = file.map(lambda x:x.split(" "))
    model = Word2Vec()
    model = model.fit(documents)
    sy = model.findSynonyms("李浩",2)
    for word,cosines in sy:
        print(word,cosines)
    model.save(sc,"w2vmodel")
    sc.stop()

word2vec_train()