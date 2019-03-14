#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
@date 2019/3/14
"""
import sys
import codecs
PTB_TRAIN_INPUT = 'E:\\work\\golaxy_job\\golaxy_job\\python_job\\pycharm\\deepml\\' \
           'tf_learn\\dataset\\simple-examples\\data\\ptb.train.txt'
PTB_TRAIN = 'ptb.train'
PTB_TEST_INPUT = 'E:\\work\\golaxy_job\\golaxy_job\\python_job\\pycharm\\deepml\\' \
           'tf_learn\\dataset\\simple-examples\\data\\ptb.test.txt'
PTB_TEST = 'ptb.test'
VOCAB_FILE = 'ptb.vocab'
with codecs.open(VOCAB_FILE,'r','utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_2_id = {k:v for k,v in zip(vocab,range(len(vocab)))}
def get_id(word):
    return word_2_id[word] if word in word_2_id else word_2_id['<unk>']

def deal(in_file,out_file):
    fin = codecs.open(in_file,'r','utf-8')
    fout = codecs.open(out_file,'w','utf-8')
    for line in fin:
        words = line.strip().split() + ['<eos>']
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)
    fin.close()
    fout.close()
deal(PTB_TRAIN_INPUT,PTB_TRAIN)
deal(PTB_TEST_INPUT,PTB_TEST)