#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/11/26 9:21
"""
import os
import sys
import numpy as np
import math
from sklearn.datasets import load_iris
# 欧式距离函数
from ml_learn.algorithm.distance import eculide
import matplotlib.pyplot as plt

def load_data():
    """
    导入iris标准数据集
    :return:
    """
    iris = load_iris()
    data = iris.data
    target = iris.target
    target_names = iris.target_names
    return data,target,target_names

class Group(object):
    """
    定义类簇的类
    """
    def __init__(self):
        self._name = ""
        self._no = None
        self._members = []
        self._center = None
    @property
    def no(self):
        return self._no
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,no):
        self._no = no
        self._name = "G"+str(self._no)
    @property
    def members(self):
        return self._members
    @members.setter
    def members(self,member):
        if member is None:
            raise TypeError("member is None,please set value")
        if isinstance(member,list):
            self.members.extend(member)
            return
        self._members.append(member)

    def clear_members(self):
        self._members = []
    @property
    def center(self):
        return self._center
    @center.setter
    def center(self,c):
        self._center = c

class KMeans(object):
    def __init__(self,k = 2):
        if (k <= 1) or (k is None):
            raise ValueError("k's num must not none and must > 1.")
        self._k = k
        # 类簇
        self._groups = self._make_groups(k)
        self._pre_mean_value = 0
        self._current_mean_value = 1

    def _make_groups(self,k):
        """
        生成类簇
        :param k:
        :return:
        """
        groups = []
        for i in range(k):
            g = Group()
            g.name = i+1
            groups.append(g)
        return groups

    def _random_x_index(self,xlen):
        indexes = np.random.randint(0,xlen,self._k).tolist()
        return indexes

    def _compute_mean_value(self):
        sum = 0
        for i in range(len(self._groups)):
            average = self._compute_members_mean(self._groups[i].members)
            self._groups[i].center = average
            sum += average
        return sum/(len(self._groups))

    def _compute_members_mean(self,members):
        np_members = np.array(members)
        average = np.average(np_members,axis=0)
        return average

    def _find_most_nearby_group(self,x):
        np_groups = np.array([group.center for group in self._groups])
        distances = eculide(x,np_groups)
        most_similarity_index = np.argmin(distances).squeeze()
        self._groups[most_similarity_index].members = x
        return most_similarity_index

    def _clear_groups_members(self):
        for group in self._groups:
            group.clear_members()

    def fit(self,X):
        rows,cols = X.shape
        # 1.首先选取k个点为初始聚类中心点
        init_indexes = self._random_x_index(rows)
        for i,index in enumerate(init_indexes):
            self._groups[i].center = X[index]
            self._groups[i].members = X[index]
        # 2.计算每个数据与聚类中心的距离，加入到最近那一个类
        while(True):
            for i in range(rows):
                #发现距离最近的group 并将数据加入至类簇中
                self._find_most_nearby_group(X[i])
            # 3.重新计算每个类簇的平均值
            # 计算各个类别的聚类中心并返回所有类簇的均值
            self._current_mean_value = self._compute_mean_value()
            epos = np.sum(self._current_mean_value-self._pre_mean_value,axis=0).squeeze()
            if epos <= 0.00001:
                break
            # 清除历史成员 并将计算得到的均值误差保存
            self._clear_groups_members()
            self._pre_mean_value = self._current_mean_value
            # 4.重复2-3的运算，直到每个类簇额均值不再发生变化
    def plot_example(self):
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_title("KMeans Iris Example")
        plt.xlabel("first dim")
        plt.ylabel("third dim")
        legends = []
        for i in range(len(self._groups)):
            group = self._groups[i]
            members = group.members
            x = [member[0] for member in members]
            y = [member[2] for member in members]
            ax.scatter(x,y,marker='o')
            legends.append(group.name)
        plt.legend(legends,loc="best")
        plt.show()

def test_kmeans():
    data,target,target_names = load_data()
    kmeans = KMeans(k=3)
    kmeans.fit(data)
    kmeans.plot_example()

class MeanShift(object):
    """
    均值漂移聚类-基于密度
    """
    def __init__(self,radius = 0.5,distance_between_groups = 2.5,max_members = math.inf):
        self._radius = radius
        self._max_members = max_members
        self._groups = []
        self._distance_between_groups = distance_between_groups

    def _find_nearst_indexes(self,xi,XX):
        if XX.shape[0] == 0:
            return []
        distances= eculide(xi,XX)
        nearst_indexes = np.where(distances <= self._distance_between_groups)[0].tolist()
        return nearst_indexes

    def _compute_mean_vector(self,xi,datas):
        return np.sum(datas-xi,axis=0)/datas.shape[0]

    def fit(self,X):
        XX = X
        while(XX.shape[0]!=0):
            # 1.从原始数据选取一个中心点及其半径周边的点 进行漂移运算
            index = np.random.randint(0,XX.shape[0],1).squeeze()
            group = Group()
            xi = XX[index]
            XX = np.delete(XX,index,axis=0) # 删除XX中的一行并重新赋值
            nearest_indexes = self._find_nearst_indexes(xi, XX)
            nearest_datas = None
            mean_vector = None
            if len(nearest_indexes) != 0:
                nearest_datas = None
                # 2.不断进行漂移，中心点达到稳定值
                epos = 1.0
                while (True):
                    nearest_datas = XX[nearest_indexes]
                    mean_vector = self._compute_mean_vector(xi,nearest_datas)
                    xi = mean_vector + xi
                    nearest_indexes = self._find_nearst_indexes(xi, XX)
                    epos = np.abs(np.sum(mean_vector))
                    if epos < 0.00001 : break
                    if len(nearest_indexes) == 0 : break
                group.members = nearest_datas.tolist()
                group.center = xi
                XX = np.delete(XX, nearest_indexes, axis=0)
            else:
                group.center = xi
            # 3.与历史类簇进行距离计算，若小于阈值则加入历史类簇，并更新类簇中心及成员
            for i in range(len(self._groups)):
                h_group = self._groups[i]
                distance = eculide(h_group.center,group.center)
                if distance <= self._distance_between_groups:
                    h_group.members = group.members
                    h_group.center = (h_group.center+group.center)/2
                else:
                    group.name = len(self._groups) + 1
                    self._groups.append(group)
                    break
            if len(self._groups) == 0:
                group.name = len(self._groups) + 1
                self._groups.append(group)
            # 4.从余下的点中重复1-3的计算，直到所有数据完成选取

    def plot_example(self):
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_title("MeanShift Iris Example")
        plt.xlabel("first dim")
        plt.ylabel("third dim")
        legends = []
        for i in range(len(self._groups)):
            group = self._groups[i]
            members = group.members
            x = [member[0] for member in members]
            y = [member[2] for member in members]
            ax.scatter(x, y, marker='o')
            legends.append(group.name)
        plt.legend(legends, loc="best")
        plt.show()

def test_meanshift():
    data,t,tn=load_data()
    ms = MeanShift(radius=0.66,distance_between_groups=1.4)
    ms.fit(data)
    ms.plot_example()