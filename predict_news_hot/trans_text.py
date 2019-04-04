#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
@date 2019/3/19
"""
from golaxy_nlp.dataload import get_key_word
import operator

def make_vocab(filename):
    """
    制作词库
    :param filename:
    :return:
    """
    vocab = {}
    with open(filename,'r',encoding='utf-8') as fin:
        line = fin.readline()
        while line:
            tmp = line.strip().split('\t')
            words = tmp[-1].split()
            for word in words:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            line = fin.readline()
    vocab_list = sorted(vocab.items(),key=operator.itemgetter(1),reverse=True)
    wout = open('vocab','a',encoding='utf-8')
    for word in vocab_list:
        wout.write(word[0]+'\n')
    wout.close()


def read_vocab():
    """
    读取词库
    :return:
    """
    vocab = {}
    import os
    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'vocab')
    with open(vocab_path, 'r', encoding='utf-8') as fin:
        for num,word in enumerate(fin.readlines()):
            vocab[word.strip()] = num
    return vocab


def get_word_list(vocab,cont,topK=20):
    keywords = get_key_word(cont, topk=topK).split()
    word_list = []
    for word in keywords:
        if word in vocab:
            word_list.append(vocab[word])
        else:
            # 未登录词补充unk标记
            word_list.append(vocab['<unk>'])
    if len(word_list) < topK:
        # 不满足长度的词汇补充pad标记
        for i in range(topK - len(word_list)):
            word_list.append(vocab['<pad>'])
    word_list = [str(w) for w in word_list]
    return word_list