#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/10/25 14:38
"""
import os
import sys
import jieba
import jieba.analyse
import jieba.posseg

def do_cut_pos(sentence):
    jieba.posseg.POSTokenizer()
    sentence_seg=jieba.posseg.cut(sentence.strip(),)
    outstr = ''
    for seg in sentence_seg:
        outstr += '{}/{},'.format(seg.word,seg.flag)
    return outstr

def test_cut(sentence):
    """
    测试分词三种模式
    :param sentence:
    :return:
    """

    #load_user_dict()
    load_stop_word()
    seg1 = jieba.cut(sentence,cut_all=False,HMM=True)
    print("精确模式 -*-\t"+'/'.join(seg1))
    seg2 = jieba.cut(sentence, cut_all=True, HMM=True)
    print("全模式 -*-\t" + '/'.join(seg2))
    seg3 = jieba.cut_for_search(sentence, HMM=True)
    print("搜索引擎模式 -*-\t" + '/'.join(seg3))

def load_user_dict():
    """
    导入相应的自定义词典，必须是UTF-8的编码
    :return:
    """
    jieba.load_userdict("./config/newdict.txt")
def load_stop_word():
    jieba.analyse.set_stop_words("./config/stopwords.txt")



def tfidf_extract(content):
    load_user_extract_file()
    keywords = jieba.analyse.extract_tags(content, topK=20, withWeight=True, allowPOS=())
    # 访问提取结果
    for item in keywords:
        # 分别为关键词和相应的权重
        print(item[0], item[1])
def textrank_extract(content):
    load_user_extract_file()
    keywords = jieba.analyse.textrank(content, topK=50, withWeight=True, allowPOS=('ns','nr','n'))
    for item in keywords:
        # 分别为关键词和相应的权重
        print(item[0], item[1])

def load_user_extract_file():
    jieba.analyse.set_idf_path("./files/idf")
    jieba.analyse.set_stop_words("./config/stopwords.txt")


sentence = "什么情况？刘德华来杭告状索赔200万 网友称干得漂亮？"

#test_cut(sentence)
print(do_cut_pos(sentence))
print(tfidf_extract(sentence))