#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/8/29 10:09
"""
import os
import sys
import pickle
import platform
import numpy as np
from gensim.models import Word2Vec
from golaxy_nlp.similarity import cosine,eculidean
ROOTPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSNAME=platform.system()
sys.path.append(ROOTPATH)

import jieba

class TfIdf(object):
    def __init__(self):
        self.tfidf = self.load_tfidf()

    def load_tfidf(self):
        with open(ROOTPATH+"/golaxy_nlp/model/tfidf", "rb") as f:
            tfidf = pickle.load(f)
        return tfidf

    def tfidf_transform(self, sentence):
        s = self.tfidf.transform([sentence])
        return s.toarray()

    def tfidf_cosine_distance(self, sen1, sen2):
        s1 = self.tfidf_transform(sen1)
        s2 = self.tfidf_transform(sen2)
        return cosine(s1, s2)

    def get_index(self, word):
        """
        获取某个词的在词库的索引值
        """
        index = self.tfidf.vocabulary_.get(word)
        if index is None:
            return -1
        else:
            return index

    def get_tfidf(self, word, tf):
        """
        获取词的tfidf权重
        """
        index = self.get_index(word)
        if index == -1:
            return 0.0
        idf = self.tfidf.idf_[index]
        return idf * tf

class WordToVec(object):
    def __init__(self):
        self.w2v = self.load_w2v()

    def load_w2v(self):
        if OSNAME.__contains__("Win"):
            filePath = ROOTPATH + "\\golaxy_nlp\\model\\model_w2v_sg_1000"
        else:
            filePath = ROOTPATH + "/golaxy_nlp/model/model_w2v_sg_1000"
        return Word2Vec.load(filePath)

    def word2vec_transform(self, sentence):
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

    def compute_cosine(self, s1, s2):
        vec1 = self.word2vec_transform(s1)
        vec2 = self.word2vec_transform(s2)
        return cosine(vec1, vec2)

    def compute_eculidean(self, s1, s2):
        vec1 = self.word2vec_transform(s1)
        vec2 = self.word2vec_transform(s2)
        return eculidean(vec1, vec2)

    def most_similarity_words(self, word, number):
        try:
            word_list = self.w2v.wv.most_similar(word)
            if (number > 10):
                return None
            result = []
            for num_tuple in word_list[:number]:
                result.append(num_tuple[0])
            return result
        except:
            return ''

    def transform_to_vec(self, words={}):
        """
        转化词组为向量形式
        :param vocabulary:
        :param words:
        :return:
        """
        result = np.zeros(len(self.w2v.wv.vocab), dtype=np.float64)
        for word in words.keys():
            value = self.w2v.wv.vocab.get(word)
            if value is not None:
                index = value.index
                result[index] = words.get(word)
        return result

    def compute_tf(self, w):
        """
        计算词频MAP
        :param w:
        :return:
        """
        wordList = w.split(" ")
        all_count = len(wordList) * 1.0
        wordMap = {}
        for word in wordList:
            if wordMap.__contains__(word):
                wordMap[word] += 1
            else:
                wordMap[word] = 1
        for word in wordMap.keys():
            wordMap[word] = wordMap.get(word) / all_count
        return wordMap,all_count

    def get_embedding(self,word):
        e = self.w2v.wv.get(word)
        return e

class Model(TfIdf,WordToVec):
    def __init__(self):
        self.name = "model"
        self.tfidf = self.load_tfidf()
        self.w2v = self.load_w2v()

    def words_extract_tfidf_vec(self,wordStrList=''):
        words,wordsLen = self.compute_tf(wordStrList)
        #统计词频信息
        wordvector = []
        for word,tf in words.items():
            tfidf_value = self.get_tfidf(word,tf*1.0/wordsLen)
            print(word,'\t',tfidf_value)
            vec = self.word2vec_transform(word)
            wordvector.append((word,tfidf_value,vec))
        return wordvector
    def sort_with_tfidf(self,words):
        pass

    def get_vec(self,words):
        vecs=[]
        indexs=[]
        for index,word in enumerate(words):
            vecs.append(word[-1])
            indexs.append(index)
        return [x[0] for x in vecs],indexs

    def cluster(self,wordStrList,K=10):
        words = self.words_extract_tfidf_vec(wordStrList)
        if K >= len(words):
            return self.sort_with_tfidf(words)
        from sklearn.cluster import KMeans
        self.kmean = KMeans(n_clusters=K)
        vecs,indexs = self.get_vec(words)
        self.kmean.fit_transform(vecs)
        labels = self.kmean.labels_
        word_map = {}
        for i,label in enumerate(labels):
            if word_map.__contains__(label):
                word_map[label].append(words[i][0])
            else:
                word_map[label] = [words[i][0]]
        print(word_map)
