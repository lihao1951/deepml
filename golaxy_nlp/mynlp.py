#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/12/14 9:19
"""
import os
import sys
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r"E:/download/dataset/stanford-corenlp-full-2016-10-31",lang='zh')
sentence = "米蛆们瞬间愣住了：总统怎能这样，这不是我们向往的司法独立的自由世界"
print(nlp.ner(sentence))
print(nlp.parse(sentence))
print(nlp.dependency_parse(sentence))