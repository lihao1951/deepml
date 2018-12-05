#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/11/30 17:50
"""
import os
import sys
from kafka import KafkaConsumer
consumer = KafkaConsumer('sim_news_doc',bootstrap_servers=['10.170.130.133:9092'],group_id='li2hao')
while(True):
    data=consumer.poll(2000)
    for x in data:
        values = data[x]
        for record in values:
            print("Topic:%s partition:%s offset:%s value:%s" % (record.topic,record.partition,
                                                                record.offset,record.value))