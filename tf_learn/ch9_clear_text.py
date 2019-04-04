#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Name ch9_clear_text
@Description
    
@Author LiHao
@Date 2019/4/2
"""
import os
import re

def if_has_html_label(cont):
    pat = re.compile('<[^>]+>',re.S)
    labels = pat.findall(cont)
    if len(labels)>=1:
        return True
    else:
        return False

class Process():
    def __init__(self,en_text,zh_text,en_text_clear,zh_text_clear,en_bag_words,zh_bag_words):
        self.en_text = en_text
        self.zh_text = zh_text
        self.en_text_clear = en_text_clear
        self.zh_text_clear = zh_text_clear
        self.en_vocab = {}
        self.zh_vocab = {}
        self.en_bag_words = en_bag_words
        self.zh_bag_words = zh_bag_words

    def clear(self,save_vocab=False):
        en_reader = open(self.en_text,'r',encoding='utf-8')
        zh_reader = open(self.zh_text,'r',encoding='utf-8')
        en_line = en_reader.readline()
        while en_line:
            if not if_has_html_label(en_line.strip()):
                pass

            en_line = en_reader.readline()