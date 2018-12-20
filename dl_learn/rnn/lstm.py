#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
RNN文本分类程序
Author LiHao
Time 2018/12/5 17:26
12/06 - 12/07
"""
import os
import sys
import tensorflow as tf
text_labels = {100:"news_story",101:"news_culture", \
              102:"news_entertainment",103:"news_sports",\
              104:"news_finance",105:"news_house",\
              106:"news_house",107:"news_car",\
              108:"news_edu",109:"news_tech",\
              110:"news_military",112:"news_travel",\
              113:"news_world",114:"stock",\
              115:"news_agriculture",116:"news_game"}

def __get_toutiao_news():
    with open('./toutiao_cat_data.txt','r',encoding='utf-8') as fp:
        lines = fp.readlines()
        news_list = []
        for line in lines:
            line = line.replace("\n","")
            news_data = line.split('_!_')
            news_id = news_data[0]
            news_label = int(news_data[1])
            news_title = news_data[3]
            news_keywords = news_data[4]
            news_tuple = (news_label,news_id,news_title,news_keywords)
            news_list.append(news_tuple)
        return news_list

def lstm_get_data():
    pass