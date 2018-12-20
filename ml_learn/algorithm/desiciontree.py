#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
ID3 C4.5：决策树相关算法实现
目前不完整，预测predict、剪枝prune未实现，train 和 TreeNode 方面有待优化
这里处理的是离散型数据值

若是需要处理连续值特征，则可以变为一个二叉节点 即CART分类回归树
Author LiHao
Time 2018/12/17 17:27
"""
import numpy as np


class TreeNode(object):
    def __init__(self):
        self.entropy = 0.0  # 熵
        self.dataset = None  # 数据集
        self.children = []  # 子节点
        self.level = 0  # 所属的树层级
        self.feature = None  # 划分数据集所用的类别
        self.value = None  # 当前的节点值


def _calShannonForLabelDict(labelCount, sum):
    """
    根据数据类别计数字典 计算当前的熵
    :param labelCount:
    :param sum:
    :return:
    """
    entropy = 0.0
    for key, value in labelCount.items():
        prob = int(value) / sum
        entropy -= prob * np.log2(prob)
    return entropy


def _calShannonEntropy(dataset):
    """
    计算香农熵
    :param dataset:最后一列是类别的数据
    :return:
    """
    X, Y = dataset.shape
    # 得到每个类别的个数
    labelCount = {}
    labels = set(dataset[:, Y - 1].tolist())
    for label in labels:
        if label not in labelCount.keys():
            label_num = np.where(dataset[:, Y - 1] == label)[0].shape[0]
            labelCount[label] = label_num
    # 计算当前数据的熵
    return _calShannonForLabelDict(labelCount, X)


def _splitDataByValue(dataset, index, value):
    ii = np.where(dataset[:, index] == value)[0].tolist()
    data = []
    for i in ii:
        data.append(dataset[i])
    return np.array(data)


class DecisionTree(object):
    """
    决策树
    """

    def __init__(self, thresh=0.1):
        self._data_num = 0
        self._label_num = 0
        self._thresh = thresh
        self._X = None
        self._tree = None
        self._node_list = []

    def prune(self):
        """
        剪枝
        :return:
        """
        pass

    def predict(self):
        """
        预测
        :return:
        """
        pass

    def train(self, X, Y):
        """
        输入数据进行训练
        :param X: numpy.ndarray shape: (m,n)
        :param Y: numpy.ndarray shape: (m,)
        :return:
        """
        if X.shape[0] == Y.shape[0]:
            YY = Y.reshape((X.shape[0], 1))
            self._X = np.concatenate([X, YY], axis=1)
        else:
            return
        root = TreeNode()
        root.level = 0
        root.dataset = self._X
        root.entropy = _calShannonEntropy(root.dataset)
        self.create_tree(root)
        self._tree = root

    def create_tree(self, node):
        dataset = node.dataset
        rows, cols = dataset.shape
        entropy = node.entropy
        if cols <= 1:
            return
        if entropy < self._thresh:
            return
        bestFeature, bestInfoGain, featureValueSet = self._getBestFeature(dataset)
        print("The best feature is %d and infoGain is %s" % (bestFeature, bestInfoGain))
        for featureValue in featureValueSet:
            newNode = TreeNode()
            subData = _splitDataByValue(dataset, bestFeature, featureValue)
            subData = np.delete(subData, obj=bestFeature, axis=1)
            newNode.dataset = subData
            newNode.level = node.level + 1
            newNode.entropy = _calShannonEntropy(subData)
            newNode.value = featureValue
            node.feature = bestFeature
            node.children.append(newNode)
            # 递归生成树结构
            self.create_tree(newNode)

    def _getBestFeature(self, dataset):
        """
        获取当前数据集中最佳特征索引值
        :param dataset:
        :return:
        """
        # 获取当前索引下的数据
        entropy = _calShannonEntropy(dataset)
        rows, cols = dataset.shape
        bestInfoGain = 0.0  # 最优信息增益
        bestFeature = -1  # 最优特征索引值
        bestInfoGainRate = 0.0  # 最优信息增益率
        featureValueSet = set()
        # 选取最优的特征属性
        for i in range(cols - 1):
            iFeatureSet = set([example for example in dataset[:, i]])
            infoGain = 0.0
            infoGainRate = 0.0
            for featureValue in iFeatureSet:
                subData = _splitDataByValue(dataset, i, featureValue)
                subEntropy = _calShannonEntropy(subData)
                subProb = (1.0 * subData.shape[0]) / dataset.shape[0]
                infoGain += subProb * subEntropy
                infoGainRate -= subProb * np.log2(subProb)
            infoGain = entropy - infoGain
            infoGainRate = infoGain / infoGainRate
            print("I %s : InfoGain %s: InfoGainRate %s" % (i, infoGain, infoGainRate))
            if infoGain > bestInfoGain:  # 这里选用了信息增益作为准则
                bestInfoGain = infoGain
                bestFeature = i
                featureValueSet = iFeatureSet
        return bestFeature, bestInfoGain, featureValueSet


def make_data():
    """
    李航博士 《统计学习方法》 P59 例子
    数据值已转化为数字形式
    :return:
    """
    x = [[1, 0, 0, 1], [1, 0, 0, 2], [1, 1, 0, 2], [1, 1, 1, 1], [1, 0, 0, 1],
         [2, 0, 0, 1], [2, 0, 0, 2], [2, 1, 1, 2], [2, 0, 1, 3], [2, 0, 1, 3],
         [3, 0, 1, 3], [3, 0, 1, 2], [3, 1, 0, 2], [3, 1, 0, 3], [3, 0, 0, 1]]
    y = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


if __name__ == "__main__":
    # x,y = make_data()
    # dt = DecisionTree()
    # dt.train(x,y)
    hp = _calShannonForLabelDict({"A":10,"B":5},15)
    print(hp)