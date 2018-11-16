#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/8/23 16:34
"""
import os
import sys
from gensim.models import Word2Vec
import numpy as np
from golaxy_nlp.similarity import cosine

class MyWordVec(object):
    """
    MyWordVec
    """
    def __init__(self):
        self.w2v = self.load_w2v()

    def load_w2v(self):
       return Word2Vec.load("./model/word2vec")

    def word2vec_transform(self,sentence):
        """
        word2vec 转化句子为向量
        :param w2v:
        :param sentence:
        :return:
        """
        size = self.w2v.layer1_size
        data = sentence.split(" ")
        length = len(data)
        vec = np.zeros(shape=(1, size), dtype=np.float32)
        for word in data:
            try:
                vec += self.w2v.wv[word]
            except:
                length -= 1
                continue
        vec = vec / length
        return vec
    def cosine_between_sentences(self,s1,s2):
        v1 = self.word2vec_transform(s1)
        v2 = self.word2vec_transform(s2)
        print(cosine(v1,v2))
