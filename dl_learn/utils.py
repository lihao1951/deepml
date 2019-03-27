#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
@date 2019/3/26
"""
import os
import pandas as pd
import numpy as np
from golaxy_nlp.dataload import get_key_word,seg
from predict_news_hot.trans_text import read_vocab
ENCODE_UTF_8 = 'utf-8'
PADDING = 15

def clean():
    toutiao_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'toutiao_cat_data.txt')
    clean_toutiao_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'toutiao')
    w = open(clean_toutiao_path,'w',encoding='utf-8')
    with open(toutiao_path,'r',encoding='utf-8') as f:
        line  = f.readline()
        while line:
            data = line.strip().split('_!_')
            id = data[0]
            label  = data[1]
            label_name = data[2]
            title = data[3]
            title_seg = seg(title)
            keywords = data[4]
            if keywords is '':
                keywords = get_key_word(title)
            else:
                keywords = ' '.join(keywords.split(','))
            cont = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(id,label,label_name,title,title_seg,keywords)
            w.write(cont)
            line = f.readline()


def turn_toutiao_to_numpy():

    vocab = read_vocab()
    toutiao_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'dataset','toutiao')
    with open(toutiao_path,'r',encoding=ENCODE_UTF_8) as f:
        line = f.readline()
        X = []
        y = []
        while line:
            values = line.strip().split('\t')
            keywords = values[-1] + ' ' + values[-2]
            klist = []
            for word in keywords.strip().split():
                if vocab.__contains__(word):
                    klist.append(vocab[word])
                else:
                    klist.append(vocab['<pad>'])
            # 超过padding的需要截断
            if len(klist)>PADDING:
                klist = klist[:PADDING]
            # 不足padding的需要补全
            while len(klist)<PADDING:
                klist.append(vocab['<pad>'])
            label = int(values[1])
            X.append(klist)
            y.append([label])
            line = f.readline()
        return np.array(X,dtype=np.int32),np.array(y,dtype=np.int32)


def split_toutiao_to_train_test():
    from sklearn.model_selection import train_test_split
    data_x,data_y = turn_toutiao_to_numpy()
    train_x,test_x,train_y,test_y = train_test_split(data_x,data_y,test_size=0.2)
    print('Train x size:{},Train y size:{}'.format(train_x.shape,train_y.shape))
    print('Test x size:{},Test y size:{}'.format(test_x.shape,test_y.shape))
    return train_x,train_y,test_x,test_y