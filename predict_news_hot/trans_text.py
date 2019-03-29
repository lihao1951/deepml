#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
@date 2019/3/19
"""
from golaxy_nlp.dataload import get_key_word
import operator
TRAIN = 'mini_train'
VALID = 'mini_valid'
TEST = 'mini_test'

def make_vocab(filename):
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
    vocab = {}
    import os
    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'vocab')
    with open(vocab_path, 'r', encoding='utf-8') as fin:
        for num,word in enumerate(fin.readlines()):
            vocab[word.strip()] = num
        vocab['<pad>'] = len(vocab)
        vocab['<eos>'] = len(vocab)
        vocab['<sos>'] = len(vocab)
    return vocab


def get_word_list(vocab,cont,topK=20):
    keywords = get_key_word(cont, topk=topK).split()
    word_list = []
    for word in keywords:
        if word in vocab:
            word_list.append(vocab[word])
        else:
            word_list.append(vocab['<pad>'])
    if len(word_list) < topK:
        for i in range(topK - len(word_list)):
            word_list.append(vocab['<pad>'])
    word_list = [str(w) for w in word_list]
    return word_list


def change_2_id_list(filename):
    vocab = read_vocab()
    wout = open(filename+'_sig', 'a', encoding='utf-8')
    with open(filename,'r',encoding='utf-8') as fin:
        line = fin.readline()
        while line:
            tmp = line.strip().split('\t')
            try:
                cont = tmp[-1]
                value = tmp[2]
            except:
                line = fin.readline()
                continue
            word_list = get_word_list(vocab,cont)
            if float(value)>=0.7:
                value = 2
            elif float(value)>=0.3:
                value = 1
            else:
                value = 0
            wline = '{}-{}\n'.format(value,' '.join(word_list))
            wout.write(wline)
            line = fin.readline()
    wout.close()