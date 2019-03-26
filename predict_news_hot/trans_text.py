#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
@date 2019/3/19
"""
# from golaxy_nlp.dataload import seg
# from golaxy_nlp.dataload import clean_sentence
from golaxy_nlp.dataload import get_key_word
import operator
TRAIN = 'mini_train'
VALID = 'mini_valid'
TEST = 'mini_test'

def merge():
    init = open('mini_train_sig_clean', 'r')
    zero = []
    one = []
    lines = init.readlines()
    for line in lines:
        label = str(line.split('-')[0])
        if label=='1':
            one.append(line)
        else:
            zero.append(line)
    roll = True
    merge = open('mini_train_sig_merge','a')
    while len(zero)!=0 and len(one)!=0:
        if roll:
            merge.write(zero.pop(-1))
        else:
            merge.write(one.pop(-1))
        roll = not roll


def clean():
    ALL=15000
    all_1 = 0
    all_0 = 0
    w = open('mini_train_sig_clean','a')
    with open('mini_train_sig','r') as f:
        line = f.readline()
        while line:
            label = str(line.split('-')[0])
            if label=='1':
                if all_1<=ALL:
                    w.write(line)
                    all_1+=1
            else:
                if all_0<=ALL:
                    w.write(line)
                    all_0 += 1
            line = f.readline()
    w.close()


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
    with open('vocab', 'r', encoding='utf-8') as fin:
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
    if len(word_list) < 20:
        for i in range(20 - len(word_list)):
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