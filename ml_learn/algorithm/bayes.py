#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/11/2 16:17
"""
import os
import sys
import numpy as np
from dataset.dataUtils import load_mnist
from ml_learn.algorithm.base import Algorithm
from sklearn.naive_bayes import GaussianNB

class NativeBayes(Algorithm):
    def __init__(self,nb_lambda = 0):
        #每个参数的概率
        self._parameters_prop = {}
        #类别名称及各个类别的概率
        self._labels_prop = {}
        #拉普拉斯平滑参数
        self._lambda = nb_lambda
        #类别前缀
        self._label_prefix = "C"
        #参数及特征前缀
        self._parameters_predix = "P"
        #训练数据总数
        self._data_num = 0
        #类别名称列表
        self._label_name_list = None
        #特征的取值集合，为map(set)嵌套
        self._features_set = {}

    def predict(self,x):
        """
        预测
        :param x:
        :return:
        """
        dx,fy = x.shape
        for i in range(dx):
            max_class_name = ""
            max_class_prop = 0.0
            for class_name,class_prop in self._labels_prop.items():
                likehood_prop = 1.0
                for j in range(fy):
                    current_data_name = self._parameters_predix + str(j) + "_" + str(x[i][j]) + "|" + class_name
                    if self._parameters_prop.get(current_data_name) is not None:
                        likehood_prop *= self._parameters_prop.get(current_data_name)
                    else:
                        likehood_prop *= 0.001
                current_prop = class_prop * likehood_prop
                if current_prop >= max_class_prop:
                    max_class_prop = current_prop
                    max_class_name = class_name
            print("The ",i,"'s data class is ",max_class_name," max prop is ",max_class_prop)

    def train(self,X,Y):
        """
        训练-贝叶斯估计方法
        :param X:
        :param Y:
        :return:
        """
        data_num,feature_num = X.shape
        self._data_num = data_num
        label_list = []
        for i in range(data_num):
            #读入类别标签
            label_name = self._label_prefix+str(Y[i])
            if self._labels_prop.__contains__(label_name):
                self._labels_prop[label_name] += 1
            else:
                self._labels_prop[label_name] = 1
                label_list.append(label_name)
            for j in range(feature_num):
                feature_name = self._parameters_predix+str(j)
                if self._features_set.__contains__(feature_name):
                    self._features_set[feature_name].add(str(X[i][j]))
                else:
                    self._features_set[feature_name] = set([str(X[i][j])])
                current_data_name = self._parameters_predix+str(j)+"_"+str(X[i][j])+"|"+label_name
                if self._parameters_prop.__contains__(current_data_name):
                    self._parameters_prop[current_data_name] += 1
                else:
                    self._parameters_prop[current_data_name] = 1
        self._label_name_list = label_list
        #求参数的概率值
        for key,value in self._parameters_prop.items():
            belong_class = key.split("|")[-1]
            belong_feature_name = key.split("_")[0]
            #数据的每个特征维度值的种类个数
            Sj = len(self._features_set[belong_feature_name])
            class_num = self._labels_prop[belong_class]
            self._parameters_prop[key] = (value*1.0+self._lambda)/(class_num+self._lambda*Sj)
        #求每个类别的概率值
        for key,value in self._labels_prop.items():
            self._labels_prop[key] = (value*1.0 + self._lambda)/(data_num + len(self._labels_prop)*self._lambda)
        print("LH -*- Train Done.")

def bayes_mnist_train():
    mnist = load_mnist()
    nbclf = NativeBayes(nb_lambda=1)
    nbclf.train(mnist.test.images,mnist.test.labels)
    nbclf.predict(mnist.test.images[2:50])
    print(mnist.test.labels[2:50])

def bayes_test():
    X = np.array([[1,1],[1,2],[1,2],[1,1],[1,1],[2,1],[2,2],[2,2],[2,3],[2,3],[3,3],[3,2],[3,2],[3,3],[3,3]])
    Y = np.array([[-1],[-1],[1],[1],[-1],[-1],[-1],[1],[1],[1],[1],[1],[1],[1],[-1]])
    T = np.array([[2,4]])
    import time
    start1 = time.time()
    clf = NativeBayes(nb_lambda=2)

    clf.train(X,Y)
    clf.predict(T)
    end1 = time.time()
    print(end1-start1)

if __name__ == '__main__':
    bayes_test()
    #bayes_mnist_train()