#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/10/12 10:28
"""
import os
import sys
import numpy as np
import random
from math import sqrt
"""
验证梯度下降程序
"""
def random_num():
    #读取50个随机数据
    r = []
    func = lambda x:x**2+2*x+1+random.uniform(-0.3,0.3)
    for i in np.arange(-5,5,0.2):
        r.append([i,func(i)])
    return r

class Gradient(object):
    """
    利用数据求解 求解一元二次方程 x**2 + 2*x + 1的系数
    分别利用
    梯度下降法/Mini-Batch梯度下降法
    动量梯度下降法
    RMSProp
    Adam
    """
    def __init__(self,alpha = 0.01, error = 1e-03,A=0.1,B=0.1,C=0.1,data_num=100,max_iter=30000):
        """
        _ 内部变量或函数，可以在子类中使用
        __内部变量或函数，子类中都无法使用
        默认有10000条数据
        :param alpha: 步长
        :param error: 误差
        :param maxcount: 最大迭代次数
        :param X: 初始值
        """
        self._alpha = alpha
        self._error = error
        #读入数据并初始化各个系数
        self.data_num = data_num
        self._data = self.inputs()
        self.A = A
        self._A = 1
        self.B = B
        self._B = 2
        self.C = C
        self._C = 1
        self._max_iter = max_iter

    def inputs(self):
        datas = []
        for i in range(self.data_num):
            datas.extend(random_num())
        #print("LH -*- 得到的数据个数：",len(datas))
        return datas

    def get_y_hat(self,mini_data):
        """
        获取预测值 y^
        :param mini_data:
        :return:
        """
        y_hat = self.A * np.power(mini_data[:,0],2) + self.B * mini_data[:, 0] + self.C
        return y_hat

    def gradient(self,mini_data,y_hat,m):
        self.A = self.A - self._alpha * np.sum((y_hat-mini_data[:,1])*np.power(mini_data[:,0],2))/m
        self.B = self.B - self._alpha * np.sum((y_hat-mini_data[:,1])*mini_data[:,0])/m
        self.C = self.C - self._alpha * np.sum(y_hat-mini_data[:,1])/m

    def minibatch_gradient_train(self,m=2,error=1e-02):
        """
        minibatch-梯度下降
        :param m:
        :param error:
        :return:
        """
        self._error = error
        all_lens = len(self._data)
        if all_lens % m ==0:
            epoch = int(all_lens/m)
        else:
            epoch = int(all_lens/m) + 1
        Error = 1.0
        count = 1
        while(Error>self._error and count < self._max_iter):
            #分批次求解
            ie = random.randint(0,epoch-1)
            mini_data = np.array(self._data[ie*m:(ie+1)*m],dtype=np.float32)
            current_m = mini_data.shape[0]
            y_hat = self.get_y_hat(mini_data)
            Error = (abs(self.A - self._A) + abs(self.B - self._B) + abs(self.C - self._C)) / 3
            #print("LH -*- epoch: ",ie,"\tloss : ",Error," A,B,C:",self.A,self.B,self.C)
            self.gradient(mini_data,y_hat,current_m)
            count += 1
        print("LH -*- Minibatch -*-Final A,B,C,iter:",self.A,self.B,self.C,count," error:",Error)

    def momentum_gradient(self,mini_data,y_hat,m,pre_va,pre_vb,pre_vc,beta):
        da =  np.sum((y_hat - mini_data[:, 1]) * np.power(mini_data[:, 0], 2)) / m
        db =  np.sum((y_hat - mini_data[:, 1]) * mini_data[:, 0]) / m
        dc =  np.sum(y_hat - mini_data[:, 1]) / m
        va = da * (1 - beta) + beta * pre_va
        vb = db * (1 - beta) + beta * pre_vb
        vc = dc * (1 - beta) + beta * pre_vc
        self.A = self.A - self._alpha * va
        self.B = self.B - self._alpha * vb
        self.C = self.C - self._alpha * vc
        return va,vb,vc

    def momentum_gradient_train(self,m=2,beta=0.9,error=1e-02):
        """
        动量梯度下降
        较之梯度下降更加快速
        相当于在梯度方向中有一个加速度，且当前步的梯度与历史的梯度方向有关
        :param m:
        :param beta:
        :param error:
        :return:
        """
        self._error = error
        all_lens = len(self._data)
        if all_lens % m == 0:
            epoch = int(all_lens / m)
        else:
            epoch = int(all_lens / m) + 1
        # 进行分批次求解
        Error = 1.0
        count = 1
        pre_va = 0.0
        pre_vb = 0.0
        pre_vc = 0.0
        while (Error >= self._error and count < self._max_iter):
            ie = random.randint(0,epoch-1)
            mini_data = np.array(self._data[ie * m:(ie + 1) * m], dtype=np.float32)
            current_m = mini_data.shape[0]
            y_hat = self.get_y_hat(mini_data)
            Error =(abs(self.A-self._A)+abs(self.B-self._B)+abs(self.C-self._C))/3
            #print("LH -*- epoch: ", ie, "\terror : ", Error, " A,B,C:", self.A, self.B, self.C)
            pre_va,pre_vb,pre_vc = self.momentum_gradient(mini_data, y_hat, current_m,pre_va,pre_vb,pre_vc,beta)
            count += 1
        print("LH -*- Momentum -*-Final A,B,C,iter:", self.A, self.B, self.C, count, " error:", Error)


    def rmsprop_gradient(self,mini_data,y_hat,m,pre_sa,pre_sb,pre_sc,beta,eps=10e-8):
        da = np.sum((y_hat - mini_data[:, 1]) * np.power(mini_data[:, 0], 2)) / m
        db = np.sum((y_hat - mini_data[:, 1]) * mini_data[:, 0]) / m
        dc = np.sum(y_hat - mini_data[:, 1]) / m
        sa = da**2 * (1 - beta) + beta * pre_sa
        sb = db**2 * (1 - beta) + beta * pre_sb
        sc = dc**2 * (1 - beta) + beta * pre_sc
        self.A = self.A - self._alpha * da / (sqrt(sa) + eps)
        self.B = self.B - self._alpha * db / (sqrt(sb) + eps)
        self.C = self.C - self._alpha * dc / (sqrt(sc) + eps)
        return sa, sb, sc

    def rmsprop_gradient_train(self,m=2,beta=0.9,error=1e-02):
        """
        RMSProp梯度下降
        自适应更新学习速率
        :param m:minibatch
        :param beta:超参数
        :param error:误差
        :return:
        """
        self._error = error
        all_lens = len(self._data)
        if all_lens % m == 0:
            epoch = int(all_lens / m)
        else:
            epoch = int(all_lens / m) + 1
        # 进行分批次求解
        Error = 1.0
        count = 1
        pre_sa = 0.0
        pre_sb = 0.0
        pre_sc = 0.0
        while (Error >= self._error and count < self._max_iter):
            ie = random.randint(0,epoch-1)
            mini_data = np.array(self._data[ie * m:(ie + 1) * m], dtype=np.float32)
            current_m = mini_data.shape[0]
            y_hat = self.get_y_hat(mini_data)
            Error = (abs(self.A - self._A) + abs(self.B - self._B) + abs(self.C - self._C)) / 3
            # print("LH -*- epoch: ", ie, "\terror : ", Error, " A,B,C:", self.A, self.B, self.C)
            pre_sa, pre_sb, pre_sc = self.rmsprop_gradient(mini_data, y_hat, current_m, pre_sa, pre_sb, pre_sc,
                                                                beta)
            count += 1
        print("LH -*- RMSProp -*- Final A,B,C,iter:", self.A, self.B, self.C, count, " error:", Error)


    def adam_gradient(self,mini_data,y_hat,m,pre_sa,pre_sb,pre_sc,pre_va,pre_vb,pre_vc,beta_1,beta_2,count,eps=10e-8):
        #correct值归一化分母
        norm_value_1 = 1-pow(beta_1,count)
        norm_value_2 = 1-pow(beta_2,count)
        #求da db dc
        da = np.sum((y_hat - mini_data[:, 1]) * np.power(mini_data[:, 0], 2)) / m
        db = np.sum((y_hat - mini_data[:, 1]) * mini_data[:, 0]) / m
        dc = np.sum(y_hat - mini_data[:, 1]) / m
        #求va vb vc
        va = da * (1 - beta_1) + beta_1 * pre_va
        vb = db * (1 - beta_1) + beta_1 * pre_vb
        vc = dc * (1 - beta_1) + beta_1 * pre_vc
        va_correct = va / norm_value_1
        vb_correct = vb / norm_value_1
        vc_correct = vc / norm_value_1
        #求sa sb sc
        sa = da ** 2 * (1 - beta_2) + beta_2 * pre_sa
        sb = db ** 2 * (1 - beta_2) + beta_2 * pre_sb
        sc = dc ** 2 * (1 - beta_2) + beta_2 * pre_sc
        sa_correct = sa / norm_value_2
        sb_correct = sb / norm_value_2
        sc_correct = sc / norm_value_2
        self.A = self.A - self._alpha * va_correct / (sqrt(sa_correct) + eps)
        self.B = self.B - self._alpha * vb_correct / (sqrt(sb_correct) + eps)
        self.C = self.C - self._alpha * vc_correct / (sqrt(sc_correct) + eps)
        return va,vb,vc,sa,sb,sc
        #return va_correct, vb_correct, vc_correct, sa_correct, sb_correct, sc_correct

    def adam_gradient_train(self,m=2,beta_1=0.9,beta_2=0.999,error=1e-02):
        """
        adam优化算法
        结合动量及RMSProp
        :param m:minibatch
        :param error:误差
        :return:
        """
        self._error = error
        all_lens = len(self._data)
        if all_lens % m == 0:
            epoch = int(all_lens / m)
        else:
            epoch = int(all_lens / m) + 1
        Error = 1.0
        count = 1
        pre_sa = 0.0
        pre_sb = 0.0
        pre_sc = 0.0
        pre_va = 0.0
        pre_vb = 0.0
        pre_vc = 0.0
        while (Error >= self._error and count < self._max_iter):
            ie = random.randint(0, epoch-1)
            mini_data = np.array(self._data[ie * m:(ie + 1) * m], dtype=np.float32)
            current_m = mini_data.shape[0]
            y_hat = self.get_y_hat(mini_data)
            Error = (abs(self.A - self._A) + abs(self.B - self._B) + abs(self.C - self._C)) / 3
            # print("LH -*- epoch: ", ie, "\terror : ", Error, " A,B,C:", self.A, self.B, self.C)
            pre_va,pre_vb,pre_vc,pre_sa, pre_sb, pre_sc = self.adam_gradient(mini_data, y_hat,current_m, pre_sa, pre_sb,pre_sc,pre_va,pre_vb,pre_vc,beta_1,beta_2,count)
            count += 1
        print("LH -*- Adam -*- Final A,B,C,iter:", self.A, self.B, self.C, count, " error:", Error)

if __name__ == '__main__':
    #random_num()
    g = Gradient()
    #g.minibatch_gradient_train(m=32,error=0.0005)
    #g.momentum_gradient_train(m=32,error=0.0005,beta=0.9)
    #g.rmsprop_gradient_train(m=32,error=0.0005,beta=0.9)
    g.adam_gradient_train(m=32,error=0.0005)