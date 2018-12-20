#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/8/23 16:34
"""
import os
import sys
from gensim.models import Word2Vec
from golaxy_nlp.dataload import clean_sentence
import numpy as np

def cosine(a,b):
    """
    余弦相似度计算
    :param a:
    :param b:
    :return:
    """
    M = np.sum(a * b)
    Z = np.linalg.norm(a) * np.linalg.norm(b)
    result = float("%.2f" % np.abs(M/Z))
    return result

class MyWordVec(object):
    """
    MyWordVec
    """
    def __init__(self):
        self.w2v = self.load_w2v()

    def load_w2v(self):
       return Word2Vec.load("./model/commentsvec")

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
        return cosine(v1,v2)

    def max_pooling(self,v,vec):
        """
        对向量做最大池化
        :param v:
        :param vec:
        :return:
        """
        length = vec.shape
        for i in range(length[0]):
            if vec.data[i] < v.data[i]:
                vec.data[i] = v.data[i]

    def window_sampling(self,vecs,window=3):
        N = len(vecs)
        window_vecs = []
        iter = N - window + 1
        for i in range(iter):
            mean_vec = 0
            for j in range(window):
                mean_vec += vecs[i + j]
            mean_vec = mean_vec / window
            window_vecs.append(mean_vec)
        return np.array(window_vecs)

    def word2vec_transform_hierachical(self,sentence):
        vecs = []
        size = self.w2v.layer1_size
        data = sentence.split(" ")
        length = len(data)
        for word in data:
            try:
                v = self.w2v.wv[word]
                vecs.append(v)
            except Exception as e:
                length -= 1
        v = self.window_sampling(vecs,window=3)
        return v

    def word2vec_transform_maxpooling(self,sentence):
        size = self.w2v.layer1_size
        data = sentence.split(" ")
        length = len(data)
        vec = np.zeros(shape=(1, size), dtype=np.float32)
        for word in data:
            try:
                v = self.w2v.wv[word]
                self.max_pooling(v,vec)
            except:
                length -= 1
        return vec

def mytest():
    wv = MyWordVec()
    a = clean_sentence("确实很不错啊")
    b = clean_sentence("点赞，值得肯定")
    print(a,'\t',b)
    d = wv.cosine_between_sentences(a,b)
    print(d)

model =Word2Vec.load('./model/cc')

def single_sim(A,B):
    global model
    size = model.layer1_size
    vec1 = np.zeros(shape=(1, size), dtype=np.float32)
    al = 0
    vec2 = np.zeros(shape=(1, size), dtype=np.float32)
    bl = 0
    for ai in A:
        try:
            vec1 += model.wv[ai]
            al += 1
        except:
            continue
    for bi in B:
        try:
            vec2 += model.wv[bi]
            bl += 1
        except:
            continue
    vec1 = vec1 / al
    vec2 = vec2 / bl
    return cosine(vec1,vec2)
a = open('./data/simi','r',encoding='utf-8')
for line in a.readlines():
    data = line.strip('\n').split(" ")
    A = data[0]
    B = data[1]
    C = data[2]
    print("simi with %s and %s real simi is %s,test is %s " % (A,B,C,single_sim(A,B)))
a.close()