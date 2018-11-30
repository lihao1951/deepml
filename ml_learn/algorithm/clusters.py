#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
聚类模块--基于numpy&python
Author LiHao
Time 2018/11/26 9:21
"""
import os
import sys
import math
import numpy as np
from sklearn.datasets import load_iris
# 欧式距离函数
from ml_learn.algorithm.distance import mydistance
from sklearn.datasets import make_moons
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
        distances = mydistance(x,np_groups)
        most_similarity_index = np.argmin(distances).squeeze()
        self._groups[most_similarity_index].members = x
        return most_similarity_index

    def _clear_groups_members(self):
        for group in self._groups:
            group.clear_members()
    def group(self):
        groups=[]
        for g in self._groups:
            for m in g.members:
                groups.append(g.name)
        return groups
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
        cxs = []
        cys = []
        for i in range(len(self._groups)):
            group = self._groups[i]
            members = group.members
            x = [member[0] for member in members]
            y = [member[1] for member in members]
            ax.scatter(x,y,marker='o')
            cx = group.center[0]
            cy = group.center[1]
            cxs.append(cx)
            cys.append(cy)
            legends.append(group.name)
        plt.scatter(cxs, cys, marker='+', c='k')
        plt.legend(legends,loc="best")
        plt.show()

def test_kmeans():
    X, Y = make_moons(n_samples=500)
    data,target,target_names = load_data()
    kmeans = KMeans(k=2)
    kmeans.fit(X)
    kmeans.plot_example()

class MeanShift(object):
    """
    均值漂移聚类-基于密度
    """
    def __init__(self,radius = 0.5,distance_between_groups = 2.5,bandwidth = 1,use_gk = True):
        self._radius = radius
        self._groups = []
        self._bandwidth = bandwidth
        self._distance_between_groups = distance_between_groups
        self._use_gk = use_gk #是否启用高斯核函数

    def _find_nearst_indexes(self,xi,XX):
        if XX.shape[0] == 0:
            return []
        distances= mydistance(xi,XX)
        nearst_indexes = np.where(distances <= self._distance_between_groups)[0].tolist()
        return nearst_indexes

    def _compute_mean_vector(self,xi,datas):
        distances = datas-xi
        if self._use_gk:
            sum1 = self.gaussian_kernel(distances)
            sum2 = sum1*(distances)
            mean_vector = np.sum(sum2,axis=0)/np.sum(sum1,axis=0)
        else:
            mean_vector = np.sum(datas - xi, axis=0) / datas.shape[0]
        return mean_vector

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
                # 有些博客说在一次漂移过程中 每个漂移点周边的点都需要纳入该类簇中，我觉得不妥，此处不是这样实现的，
                # 只把稳定点周边的数据纳入该类簇中
                group.members = nearest_datas.tolist()
                group.center = xi
                XX = np.delete(XX, nearest_indexes, axis=0)
            else:
                group.center = xi
            # 3.与历史类簇进行距离计算，若小于阈值则加入历史类簇，并更新类簇中心及成员
            for i in range(len(self._groups)):
                h_group = self._groups[i]
                distance = mydistance(h_group.center,group.center)
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
        cxs = []
        cys = []
        for i in range(len(self._groups)):
            group = self._groups[i]
            members = group.members
            x = [member[0] for member in members]
            y = [member[1] for member in members]
            cx = group.center[0]
            cy = group.center[1]
            cxs.append(cx)
            cys.append(cy)
            ax.scatter(x, y, marker='o')
            #ax.scatter(cx,cy,marker='+',c='r')
            legends.append(group.name)
        plt.scatter(cxs,cys,marker='+',c='k')
        plt.legend(legends, loc="best")
        plt.show()

    def gaussian_kernel(self,distances):
        """
        高斯核函数
        :param distances:
        :param h:
        :return:
        """
        left = 1/(self._bandwidth*np.sqrt(2*np.pi))
        right = np.exp(-np.power(distances,2)/(2*np.power(self._bandwidth,2)))
        return left*right

def test_meanshift(use_gk = True):
    X, Y = make_moons(n_samples=500)
    ms = MeanShift(radius=0.051,distance_between_groups=1,use_gk=use_gk)
    ms.fit(X)
    ms.plot_example()
class KernelPoint(object):
    """
    核心对象类
    """
    def __init__(self,center,nearbours):
        self._center = center
        self.nearbours = nearbours
    @property
    def center(self):
        return self._center
    @property
    def nearbours(self):
        return self.nearbours
    @nearbours.setter
    def nearbours(self,n):
        if isinstance(n,list):
            self.nearbours.append(n)

class Dbscan(object):
    """
    dbscan- Density-Based Spatial Clustering of Application with Noise
    基于密度的噪声应用空间聚类
    """
    def __init__(self,epos=0.1,minpts=5):
        self._epos = epos # 邻域半径范围
        self._minpts = minpts # 邻域内最小数据个数
        self._groups = [] #类簇集合
        self._kernel_points = [] #核心对象集合
        self._X = {} # 转化后的数据
    @property
    def group(self):
        return self._groups
    def _find_nearbours(self,xi,XX):
        """
        查找满足邻域值大小的数据索引列表
        :param xi:
        :param XX:
        :return:
        """
        if XX.shape[0] == 0:
            return []
        distances= mydistance(xi,XX)
        nearst_indexes = np.where(distances <= self._epos)[0].tolist()
        return nearst_indexes

    def _compat_X(self,X):
        """
        转化输入数据格式
        为了方便删除、存取操作
        :param X:
        :return:
        """
        rows,_ = X.shape
        CX = {}
        for row in range(rows):
            CX[row]=[False,X[row]]
        self._X = CX

    def _get_data(self,index):
        """
        获取索引的数据
        :param index:
        :return:
        """
        if isinstance(index,int):
            data = self._X.get(index)[-1]
        if isinstance(index,list):
            data = []
            for i in index:
                data.append(self._X.get(i)[-1])
        return data

    def _delete_data(self,indexes):
        """
        删除索引对应的数据
        :param indexes:
        :return:
        """
        if isinstance(indexes,int):
            self._X.get(indexes)[0] = True
        if isinstance(indexes, list):
            for index in indexes:
                self._X.get(index)[0] = True

    def _get_live_data(self):
        """
        获取未被加入到类簇集合的数据
        :return:
        """
        live_data = {}
        for key,value in self._X.items():
            if value[0]==False:
                live_data[key] = value
        return live_data

    def fit(self,X):
        """
        聚类主体
        :param X:
        :return:
        """
        self._compat_X(X)#组合X的记录
        # 1.首先遍历数据集 找到所有的核心对象
        rows,_ = X.shape
        for i in range(rows):
            # 找到小于邻域参数的数据
            nearbours = self._find_nearbours(X[i],X)
            if len(nearbours)>=self._minpts:
                self._kernel_points.append(i)

        if len(self._kernel_points) == 0 :
            # 若没有核心对象，那么每个点都为单独的一类
            for i in range(rows):
                g = Group()
                g.name = i+1
                g.members = X[i]
                g.center = X[i]
                self._groups.append(g)
            return
        while(True):
            if len(self._kernel_points) == 0:
                break
            # 2.4直到当前核心对象集合为空
            # 2.从核心对象集合中取出一个核心对象，完成对一个类簇的生成
            init_index = int(np.random.randint(0,len(self._kernel_points),size=1).squeeze())
            kernel_points = self._kernel_points[init_index]
            self._kernel_points.remove(kernel_points)
            # 2.1拿取第一个核心对象生成类簇，加入当前核心对象集合
            current_points = set()  # 当前簇的样本集合
            current_kernel_points = set()
            g = Group()
            g.center = X[kernel_points]
            self._delete_data(kernel_points)
            current_kernel_points.add(kernel_points)
            delete_kernel_points = set()
            while len(current_kernel_points)!= 0:
                # 2.2然后当前核心对象集合中取一个核心点 找到该核心对象的所有邻域点，则邻域点即为该类簇的成员数据
                current_point_index = current_kernel_points.pop()
                current_points.add(current_point_index)
                delete_kernel_points.add(current_point_index)
                nearbours_points_indexes = self._find_nearbours(X[current_point_index],X)
                current_points = current_points.union(set(nearbours_points_indexes))
                # 2.3用这些成员与原始核心对象集合做交集，若有重合则将其加入当前核心对象集合中，重复上述查询邻域过程
                union_kernel_points = current_points.intersection(set(self._kernel_points))
                current_kernel_points = current_kernel_points.union(union_kernel_points)
                current_kernel_points = current_kernel_points.difference(delete_kernel_points)
                # 3 重复上述步骤
            current_datas = self._get_data(list(current_points))
            self._delete_data(list(current_points))
            g.members = current_datas
            g.name = len(self._groups) + 1
            self._groups.append(g)
            s = set(self._kernel_points)
            s = s.difference(current_points)
            self._kernel_points = list(s)
        # 查询现有余下的数据 也就是噪声点
        live_datas = self._get_live_data()
        for key ,value in live_datas.items():
            g = Group()
            g.name = len(self._groups) + 1
            g.center = value[-1]
            g.members = value[-1]
            self._groups.append(g)

    def plot_example(self):
        """
        画图
        """
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_title("Dbscan Iris Example")
        plt.xlabel("first dim")
        plt.ylabel("third dim")
        legends = []
        cxs = []
        cys = []
        for i in range(len(self._groups)):
            group = self._groups[i]
            members = group.members
            x = [member[0] for member in members]
            y = [member[1] for member in members]
            cx = group.center[0]
            cy = group.center[1]
            cxs.append(cx)
            cys.append(cy)
            ax.scatter(x, y, marker='o')
            #ax.scatter(cx,cy,marker='+',c='r')
            legends.append(group.name)
        plt.scatter(cxs,cys,marker='+',c='k')
        plt.legend(legends, loc="best")
        plt.show()

def test_dbscan():
    # X=np.array([[1,2],[2,1],[2,3],[4,3],[5,8],[6,7],[6,9],[7,9],[9,5],[1,12],[3,12],[5,12],[3,3]])
    X,Y=make_moons(n_samples=500)
    #data,t,tname = load_data()
    dbscan = Dbscan(0.1,3)
    dbscan.fit(X)
    dbscan.plot_example()

class Point(object):
    """
    数据点类
    """
    def __init__(self,index,data,is_core = False,coredistance = math.inf,reachdistance = math.inf,belongCoreIndex = None):
        self._index = index
        self._data = data
        self._core_distance = coredistance
        self._reach_distance = reachdistance
        self._belong_core_index = belongCoreIndex
        self._is_core = is_core
        self._members = []

    @property
    def members(self):
        return self._members
    @members.setter
    def members(self,b):
        if isinstance(b,int):
            self._members.append(b)
        if isinstance(b,list):
            self._members.extend(b)
    @property
    def core(self):
        return self._is_core
    @core.setter
    def core(self,c):
        if isinstance(c,bool):
            self._is_core = c
    @property
    def index(self):
        return self._index
    @index.setter
    def index(self,i):
        self._index = i
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self,d):
        self._data = d
    @property
    def coredistance(self):
        return self._core_distance
    @coredistance.setter
    def coredistance(self,c):
        self._core_distance = c
    @property
    def reachdistance(self):
        return self._reach_distance
    @reachdistance.setter
    def reachdistance(self,r):
        self._reach_distance = r
    @property
    def belongcoreindex(self):
        return self._belong_core_index
    @belongcoreindex.setter
    def belongcoreindex(self,b):
        self._belong_core_index = b

class Optics(object):
    def __init__(self,epsilon=0.5,minpts=3):
        self._epsilon = epsilon
        self._minpts = minpts
        self._X = {}
        self._groups = []
        self._sort_ponits = []
        self._kernel_points_index = set()
        self._result_points = []
        self._is_visited = {}

    def _init_data_points(self,X):
        rows, _ = X.shape
        for i in range(rows):
            xi = X[i]
            distances = mydistance(xi,X)
            dis_num = np.where(distances<=self._epsilon)[0].shape[0]
            include_points_index = distances.argsort()[1:dis_num].tolist()
            #若是核心点
            if dis_num >= self._minpts:
                point = Point(index=i,data=xi,is_core=True,belongCoreIndex=i)
                point.members = include_points_index
                cored = distances[include_points_index[1]]
                point.coredistance = cored
                self._kernel_points_index.add(i) #加入核心点集合
            else:
                point = Point(index=i,data=xi,is_core=False)
            self._X[i] = point
            self._is_visited[i] = False
        return rows

    def fit(self,X):
        rows = self._init_data_points(X)
        while(len(self._X)!=0):

            i = self._kernel_points_index.pop()
            kernel_point = self._X.pop(i)
            self._result_points.append(kernel_point)
            self._is_visited[i] = True
            members_list = kernel_point.members
            for mem in members_list:
                self._sort_ponits.append(mem)
                self._is_visited[mem] = True
                self._X.pop(mem)
            while(len(self._sort_ponits)!=0):
                pass

class Birch(object):
    pass

class GMM(object):
    pass

class Spectral(object):
    pass

def plot_sklearn_example(data,labels,name="Birch"):
    cluster_index = {}
    for i in range(labels.shape[0]):
        if cluster_index.__contains__(labels[i]):
            cluster_index[labels[i]].append(i)
        else:
            cluster_index[labels[i]] = [i]
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.set_title(name+" iris examples")
    G = 'G'
    legend_group = []
    for key, values in cluster_index.items():
        x = []
        y = []
        legend_group.append(G+str(key))
        for value in values:
            x.append(data[value][0])
            y.append(data[value][1])
        ax.scatter(x, y, marker='o')
    ax.legend(legend_group,loc="best")
    plt.show()

def test_birch():
    """
    brich适合低维度的数据集/不适合维度大于20
    :return:
    """
    moon = make_moons(n_samples=200)[0]
    from sklearn.cluster import Birch
    b = Birch(threshold=1,n_clusters=None)
    b.fit(moon)
    # print(b.labels_)
    labels = b.labels_
    plot_sklearn_example(moon,labels)

def test_spectrcal():
    """
    谱聚类测试
    :return:
    """
    from sklearn.cluster import SpectralClustering

    moon = make_moons(n_samples=200)[0]
    s = SpectralClustering(n_clusters=2)
    s.fit(moon)
    labels = s.labels_
    plot_sklearn_example(moon,labels,name="Spectral")
def test_AffinityPropagation():
    """
    AP聚类算法是基于数据点间的"信息传递"的一种聚类算法
    :return: 
    """
    from sklearn.cluster import AffinityPropagation
    ap = AffinityPropagation(damping=0.81,convergence_iter=2)
    moon = make_moons(n_samples=200)[0]
    ap.fit(moon)
    labels = ap.labels_
    plot_sklearn_example(moon,labels,name="AP")

class SinglePass(object):
    def __init__(self,thresh=1):
        self._X = None
        self._groups = []
        self._thresh = thresh

    def _find_most_near_index(self):
        pass

    def fit(self,X):
        self._X = X
        rows,cols = X.shape
        for row in range(rows):
            xi = X[row]
            final_group = None
            for group in self._groups:
                d = mydistance(xi,group.center,isEculidean=False)
                if d <= self._thresh:
                    final_group = group
            if final_group == None:
                g = Group()
                g.name = len(self._groups) +1
                g.members = xi
                g.center = xi
                self._groups.append(g)
            else:
                g.members = xi

    def plot_example(self):
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_title("SinglePass Moon Example")
        plt.xlabel("first dim")
        plt.ylabel("third dim")
        legends = []
        cxs = []
        cys = []
        for i in range(len(self._groups)):
            group = self._groups[i]
            members = group.members
            x = [member[0] for member in members]
            y = [member[2] for member in members]
            ax.scatter(x,y,marker='o')
            cx = group.center[0]
            cy = group.center[2]
            cxs.append(cx)
            cys.append(cy)
            legends.append(group.name)
        plt.scatter(cxs, cys, marker='+', c='k')
        plt.legend(legends,loc="best")
        plt.show()


def test_singlePass():
    moon = make_moons(n_samples=200)[0]
    data,t,tn = load_data()
    sg = SinglePass(thresh=0.01)
    sg.fit(data)
    sg.plot_example()
test_singlePass()
