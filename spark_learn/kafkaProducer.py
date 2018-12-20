#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/11/30 17:37
"""
import os
import sys
from kafka import KafkaProducer
import time
# 生产者
topic = "sim_news_doc"
producer = KafkaProducer(bootstrap_servers="10.170.130.133:9092")
count = 1
while(True):
    producer.send(topic,b'2')
    count += 1
    print(count)
    time.sleep(1)