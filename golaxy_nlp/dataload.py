#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/8/2 13:51
"""
import os
import sys
import re
import pymongo
import jieba
import platform
from golaxy_nlp.regex import is_num
from golaxy_nlp.regex import remove_illegal_mark

# 加载自定义词典
#jieba.load_userdict('./config/newdict.txt')

def _get_os_name():
    """
    返回当前所在的系统名称
    :return:
    """
    os_name = platform.platform()
    if os_name.lower().__contains__("windows"):
        return "win"
    else:
        return "other"

def _get_current_path():
    """
    获取当前文件夹所在的绝对路径
    :return:
    """
    filepath = os.path.realpath(__file__)
    dirpath = os.path.dirname(filepath)
    return dirpath

def getStopWords():
    """
    获取停用词
    :return:
    """
    os_name = _get_os_name()
    dirpath = _get_current_path()
    if os_name == "win":
        path = dirpath+"\\config\\stopwords.txt"
    else:
        path = dirpath + "/config/stopwords.txt"
    with open(path,'r',encoding="UTF-8") as f:
        stop_lines = f.read().splitlines()
    return stop_lines

def _close_mongo(client):
    """
    关闭mongo连接
    :return:
    """
    if client is not None:
        client.close()

def _build_mongo_connect(ip="10.18.17.209",port=27017):
    """
    连接mongo，返回客户端
    :param ip:
    :param port:
    :return:
    """
    client = pymongo.MongoClient(host=ip,port=port)
    return client

def _seg_sentence(sentence):
    """
    分割文本
    :param sentence:
    :return:
    """
    sentence_seged = jieba.cut(sentence.strip())
    outstr = ''
    stopwords = getStopWords()
    for word in sentence_seged:
        if len(word)==1:
            continue
        if is_num(word):
            continue
        if word not in stopwords:
            outstr += word + ' '
    return outstr.strip().split(' ')

def validation_data(tfidf=False,count = 0):
    FILE_NAME = './data/all_news.txt'
    """
    获取验证数据集
    :return:
    """
    # 若存在数据
    if os.path.exists(FILE_NAME+"5"):
        r = []
        with open(FILE_NAME,encoding="utf-8") as f:
            count = 1
            while True:
                line = f.readline()
                if len(line) == 0 :break
                if tfidf:
                    data = line.replace("\n","")
                else:
                    data = line.replace("\n","")
                count += 1
                # if count>200:break
                r.append(data)
        print("读取数据完成...")
        return r
    else:
        client = _build_mongo_connect()
        db = client.get_database("yq")
        col = db.get_collection("news_train_data")
        skips = count
        datas = col.find().batch_size(200).skip(skips)
        # r为 [[],[],...,[]] 类型的数据
        r = []
        with open(FILE_NAME,"a",encoding="utf-8") as w:
            for data in datas:
                # 清洗数据并分词
                count += 1
                if data['content'] is None:continue
                content = remove_illegal_mark(data['content'])
                clear_content = _seg_sentence(content)
                print(count,'\t')
                w.write(' '.join(clear_content)+"\n")
                r.append(clear_content)
        _close_mongo(client)
        return r

def get_comments():
    """
    存储评论数据
    :return:
    """
    FILE_NAME = "./data/all_comments.txt"
    client = _build_mongo_connect()
    db = client.get_database("yq")
    col = db.get_collection("comment_train_data")
    datas = col.find().batch_size(200).limit(10000)
    # r为 [[],[],...,[]] 类型的数据
    count = 1
    with open(FILE_NAME, "a", encoding="utf-8") as w:
        for data in datas:
            # 清洗数据并分词
            content = remove_illegal_mark(data['content'])
            c_content = jieba.cut(content)
            print(count, '\t')
            w.write(' '.join(c_content).strip() + "\n")
            count += 1
    _close_mongo(client)

def re_split(content):
    """
    多模式切分字符串
    :param content:
    :return:
    """
    return re.split('[_,，。；]',content)

def clean_sentence(sentence):
    """
    对文本数据清洗并分词
    :param sentence:
    :return:
    """
    content = remove_illegal_mark(sentence)
    clear_content = _seg_sentence(content)
    return  ' '.join(clear_content)