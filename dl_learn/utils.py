#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
@date 2019/3/26
"""
import os
import pandas as pd
from golaxy_nlp.dataload import get_key_word,seg


def split_toutiao():
    toutiao_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'dataset','toutiao_cat_data.txt')
    print('toutiao path : {}'.format(toutiao_path))
    table = pd.read_table(toutiao_path,sep='_!_',header=None,names=['id','label','label_name','title','keyword'],encoding='utf-8')
    return table


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
clean()