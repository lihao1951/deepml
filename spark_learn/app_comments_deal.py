#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/12/10 16:33
"""
import os
import sys
from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils,TopicAndPartition
import json

def assemble_kafka_setup(broker_list):
    """
    组装kafka的配置
    :param broker_list:
    :return:
    """
    kafkaParams = {"metadata.broker.list":broker_list}
    return kafkaParams

def deal_rdd_save_local(rdd):
    comments = rdd.collect()
    save_path = r"/home/hadoop/lihao/appcomments"
    sum = 0
    with open(save_path,'a',encoding='utf-8') as fp:
        for comment in comments:
            mycomment = json.loads(comment[-1])
            body = mycomment['cont']
            if body is "" : continue
            sum += 1
            print(body)
            fp.write(body.replace("\n","")+"\n")
    print("All sum is :%d" % sum)

def connect_kafka(broker_list="",topic_list=[],query_time=10):
    kafkaParams = assemble_kafka_setup(broker_list)
    sc = SparkContext(master="local[2]", appName="AppComments")
    ssc = StreamingContext(sc, query_time)
    kvs = KafkaUtils.createDirectStream(ssc=ssc, kafkaParams=kafkaParams, topics=topic_list)
    kvs.foreachRDD(lambda x: deal_rdd_save_local(x))
    ssc.start()
    ssc.awaitTermination()
    ssc.stop()
    sc.stop()

if __name__ == '__main__':
    #connect_kafka('10.170.130.133:9092',['base_yt_app'])
    l = " 👍 👍 "
    print(l)


