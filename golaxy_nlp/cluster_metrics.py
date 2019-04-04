#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Name cluster_metrics
@Description
    聚类评估方法
@Author LiHao
@Date 2019/4/4
"""
import sys
import math

def _find_dict_and_dataset(p_class = [],c_class = []):
    """
    把标注类簇p_class和聚类后的类簇c_class
    转化成{doc:"类标"}形式并得到全部数据集X
    示例输入：[["1","2"],["3","4"]],[["1","3"],["2","4"]]
    :param p_class:
    :param c_class:
    :return: X,p_map,c_map
    """
    X = []
    p_map = {}
    c_map = {}
    for pi in range(len(p_class)):
        for i in range(len(p_class[pi])):
            doc = p_class[pi][i]
            p_map[doc] = "P"+str(pi)
            X.append(doc)

    for ci in range(len(c_class)):
        for i in range(len(c_class[ci])):
            doc = c_class[ci][i]
            c_map[doc] = "C"+str(ci)
    return X,p_map,c_map

def TraditionalIndexRate(p_class = [],c_class = []):
    """
    输出传统统计指标,在文本聚类中运用不多
    示例输入：[["1","2"],["3","4"]],[["1","3"],["2","4"]]
    :param p_class:
    :param c_class:
    :return: R,J,FM,PA,NA,AA
    """
    if len(p_class) == 0 or len(c_class) == 0 :
        print("输入的历史数据标签是空")
        sys.exit(1)
    X,p_map,c_map = _find_dict_and_dataset(p_class,c_class)
    SS = 0
    SD = 0
    DS = 0
    DD = 0
    #遍历数据，根据相等关系 计算SS、SD、DS、DD
    for xi in range(len(X)-1):
        for xj in range(xi+1,len(X)):
               if c_map.get(X[xi]) == c_map.get(X[xj]):
                   if p_map.get(X[xi]) == p_map.get(X[xj]):
                       SS += 1
                   else:
                       SD += 1
               else:
                   if p_map.get(X[xi]) == p_map.get(X[xj]):
                       DS += 1
                   else:
                       DD += 1
    M = SS + DS + SD + DD
    #输出一些重要指标(传统聚类统计指标)
    R = (SS + DD) * 1.0 / M
    J = SS * 1.0 / (SS + SD + DS)
    FM = math.sqrt((SS*1.0/(SS+SD))*(SS*1.0/(SS+DS)))
    print("(R) Rand Statistic: %f" % R)
    print("(J) Jaccard coefficient: %f" % J)
    print("(FM) Fowlkes and Mallows index: %f" % FM)

    PA = SS * 1.0 / (SS + DS)
    NA = DD * 1.0 / (DD + SD)
    AA = (PA + NA) / 2
    print("(PA) Positive accuracy: %f" % PA)
    print("(NA) Negative accuracy: %f" % NA)
    print("(AA) Averaged accuracy: %f" % AA)
    return R,J,FM,PA,NA,AA

def BasedOnManualAnnotationIndexRate(p_class = [],c_class = []):
    """
    基于人工标注类的准确率、召回率、F值进行计算，返回Class_F值，是一个整体指标，推荐使用
    示例输入：[["1","2"],["3","4"]],[["1","3"],["2","4"]]
    :param p_class:
    :param c_class:
    :return: Class_F
    """
    FP = []
    for j in range(len(p_class)):
        Pj = set(p_class[j])
        FPj = []
        for i in range(len(c_class)):
            Ci = set(c_class[i])
            pre_ji = len(Pj & Ci) * 1.0 / len(Ci)
            rec_ji = len(Pj & Ci) * 1.0 / len(Pj)
            Fji = 0.0
            if pre_ji + rec_ji != 0:
                Fji = 2 * pre_ji * rec_ji / (pre_ji + rec_ji)
            FPj.append(Fji)
        FP.append(max(FPj))
    P = 0.0
    PFP = 0.0
    for j in range(len(p_class)):
        Pj_len = len(p_class[j])
        P += Pj_len
        PFP += Pj_len * FP[j]
    Class_F = PFP /  P
    return Class_F

def BasedOnClusterIndexRate():
    """
    基于簇的准确率、召回率及F值，较为不常用，这里等待以后实现
    """
    pass

def _find_doc_index(doc="",mmap={},C=""):
    """
    发现文档在类别中的索引值
    :param doc:
    :param mmap:
    :param C:
    :return:
    """
    label = mmap.get(doc)
    num = int(label.replace(C,""))
    return num

def BasedOnDocIndexRate(p_class = [],c_class = []):
    """
    基于文档的准确率及召回率及F值，最后的结果为全部文档的平均值，该值与Class_F想类似，可以作为文本聚类效果的评价
    示例输入：[["1","2"],["3","4"]],[["1","3"],["2","4"]]
    :param p_class:
    :param c_class:
    :return: P,R,F
    """
    X, p_map, c_map = _find_dict_and_dataset(p_class, c_class)
    Precision = []
    Recall = []
    for doc in X:
        p_index = _find_doc_index(doc,p_map,"P")
        c_index = _find_doc_index(doc, c_map, "C")
        S_correct = set(p_class[p_index])
        S_compute = set(c_class[c_index])
        p = len(S_correct & S_compute) * 1.0 / len(S_compute)
        r = len(S_correct & S_compute) * 1.0 /len(S_correct)
        Precision.append(p)
        Recall.append(r)
    P_aver = sum(Precision) / len(Precision)
    R_aver = sum(Recall) / len(Recall)
    F = 2 * P_aver * R_aver / (P_aver + R_aver)
    return P_aver,R_aver,F

def EntropyIndexRate(p_class = [],c_class = []):
    """
    计算聚类结果的熵大小-推荐使用的指标
    但是熵倾向于小类别，所以需要与Class_F等方法一起使用
    示例输入：[["1","2"],["3","4"]],[["1","3"],["2","4"]]
    输出为 c_class 长度的双精度值列表
    :param p_class:
    :param c_class:
    :return: E
    """
    E = []
    S = len(p_class)
    for i in range(len(c_class)):
        Ci = set(c_class[i])
        Ei = 0.0
        for j in range(len(p_class)):
            Pj = set(p_class[j])
            Uj = Ci & Pj
            ULog = 1.0
            if len(Uj) != 0:
                ULog = math.log(len(Ci) * 1.0 / len(Uj))
            Ei += (len(Uj) * 1.0 / len(Ci)) * ULog
        E.append(Ei / S)
    return E,math.fsum(E)/len(E)