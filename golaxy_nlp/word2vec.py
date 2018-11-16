#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/8/2 13:43
"""
import os
import sys
import gensim
import multiprocessing
from gensim.models.word2vec import LineSentence
def getPackagePath():
    path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(path))
sys.path.append(getPackagePath())
from golaxy_nlp.dataload import validation_data

def toListStr(array):
    p = ''
    for value in array:
        p += str(float(value)) + " "
    return p

def model_train_word2vec():
    """
    训练word2vec
    :return:
    """
    #sentence=validation_data()
    model = gensim.models.Word2Vec(sentences=LineSentence("./data/news_train.txt"),size=300,window=5,min_count=5,
                                   max_vocab_size=100000,workers=multiprocessing.cpu_count(),sg=1,negative=10,iter=15)
    model.save('./model/word2vec')

def save_word2vec():
    """
    存储word2vec词向量
    :return:
    """
    model = gensim.models.Word2Vec.load('word2vec_model')
    vocabulary = {}
    with open("vocabulary_wb","wb") as w:
        for x in model.wv.vocab:
            vocabulary[x] = model.wv[x]
            line = str(x)+"\t"+toListStr(vocabulary[x])+"\n"
            w.write(line.encode("utf-8"))
        print(len(vocabulary))