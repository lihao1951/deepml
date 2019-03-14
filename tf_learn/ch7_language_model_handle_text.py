#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
语言模型测试
处理英文PTB数据集文本
@author LiHao
@date 2019/3/14
"""
# 编码模块 可进行文件读取
import codecs
import collections
from operator import itemgetter
RAW_DATA = 'E:\\work\\golaxy_job\\golaxy_job\\python_job\\pycharm\\deepml\\' \
           'tf_learn\\dataset\\simple-examples\\data\\ptb.train.txt'
VOCAB_OUTPUT = 'ptb.vocab'
counter = collections.Counter()
with codecs.open(RAW_DATA,'r','utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1
sorted_word_to_cnt = sorted(counter.items(),key=itemgetter(1),reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]
sorted_words = ["<eos>"] + sorted_words

with codecs.open(VOCAB_OUTPUT,'w','utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word+'\n')