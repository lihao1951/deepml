#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/12/11 14:16
"""
import os
import sys
from pyspark.mllib.feature import Word2Vec,Word2VecModel
from pyspark import SparkContext,SQLContext

def train():
    sc = SparkContext(master="local[2]",appName="trainWord2vecApp")
    sqlContext = SQLContext(sc)
    apps = sc.textFile("./appcomments.txt")
    documents = apps.filter(lambda line:len(line)>20)
    # train_set = documents.map(lambda line:line.split(" "))
    # word2vec =Word2Vec()
    # word2vec.setVectorSize(200) #词向量大小
    # word2vec.setMinCount(3) #出现最小次数
    # word2vec.setWindowSize(3) #窗口大小
    # word2vec.setNumPartitions(5)
    # word2vec.setNumIterations(3)
    # model = word2vec.fit(train_set)
    # synon = model.findSynonyms("国家",30)
    # for k,v in synon:
    #     print(k,'\t',v)
    # model.save(sc,path="./word2vec")
    # documents.foreach(print)
    documents.saveAsTextFile("./dealappcoments")
    sc.stop()

def read_model():
    sc = SparkContext(appName="word2vecApp")
    model = Word2VecModel.load(sc,"hdfs://user/hadoop/word2vec/")
    synon = model.findSynonyms("国家",30)
    for k,v in synon:
        print(k,'\t',v)
    sc.stop()
train()