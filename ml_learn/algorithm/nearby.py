#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/10/16 9:47
"""
import os
import sys

import numpy as np

from dataset.dataUtils import load_mnist
from ml_learn.algorithm.base import Algorithm

class KNN(Algorithm):
    """
    KNN分类算法
    """
    def __init__(self):
        self._X = np.array([])
        self._Y = np.array([])

    def _distance(self,A,B):
        """
        计算数据之间的距离
        :param A:
        :param B:
        :return:
        """
        return np.sqrt(np.sum(np.power(B-A.reshape((1,B.shape[1])),2),axis=1))

    def _sort_index(self,K,l=[]):
        """
        返回前K个相似的索引
        :param K:
        :param l:
        :return:
        """
        l_index = np.argsort(l)[:K]#argsort返回排序后的索引
        return l_index

    def transfor_to_label(self,y):
        return int(np.squeeze(np.where(y==1.0)[0]))


    def predict(self,XX,K=5,y_len=10):
        """
        :param XX:
        :param K:
        :param y_len: 默认是MNIST的one-hot类别 索引值代表了数字类别
        :return:
        """
        data_num,feature_num = XX.shape
        labels = np.zeros((data_num,y_len),dtype=np.float32)
        for x in range(data_num):
            distance = self._distance(XX[x],self._X)
            dis_index = self._sort_index(K,distance)
            labelDict = {}
            predict_label = np.zeros((1,y_len),dtype=np.float32)
            for di in dis_index:
                k_label = self.transfor_to_label(self._Y[di])
                if labelDict.__contains__(k_label):
                    labelDict[k_label] += 1
                else:
                    labelDict[k_label] = 1
            predict_label[0,sorted(labelDict.items(), key=lambda x: x[1], reverse=True)[0][0]] = 1.0
            labels[x] = predict_label
        return labels

    def inputs(self,X,Y):
        self._X = X
        self._Y = Y

    def train(self,X,Y):
        self.inputs(X,Y)

def knn_train_mnist(step=50):
    knn = KNN()
    mnist = load_mnist()
    train_mnist_x = mnist.train.images
    train_mnist_y = mnist.train.labels
    test_mnist_x = mnist.test.images
    test_mnist_y = mnist.test.labels
    END = mnist.test.labels.shape[0]
    knn.train(train_mnist_x, train_mnist_y)
    all_wrong_count = 0
    for iter in np.arange(0,END,step):
        end = iter+step
        predict_labels = knn.predict(test_mnist_x[iter:end])
        predict_indexes = np.argmax(predict_labels, axis=1)
        real_indexes = np.argmax(test_mnist_y[iter:end], axis=1)
        differ_indexes = predict_indexes - real_indexes
        wrong_count = len(np.where(differ_indexes>0)[0].tolist())
        all_wrong_count += wrong_count
        wrong_rate = wrong_count*1.0/(step)
        print(iter/step," time\twrong rate is: ",wrong_rate)
    print("-*- All wrong rate is:",all_wrong_count*1.0/END)

if __name__ == '__main__':
    knn_train_mnist()