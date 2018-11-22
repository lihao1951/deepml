#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Kafka and spark test
Author LiHao
Time 2018/11/22 9:14
"""
import os
import sys
import json
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils,TopicAndPartition

offsetRanges = []
TopicName = 'news'
OFFSETFILE = TopicName+'_offset'

def store_offset_ranges(rdd):
    global offsetRanges
    offsetRanges = rdd.offsetRanges()
    return rdd

def save_offset_ranges(rdd):
    """
    存储offset信息
    :param rdd:
    :return:
    """
    root_path = os.path.dirname(os.path.realpath(__file__))
    record_path = os.path.join(root_path, OFFSETFILE)
    data = dict()
    with open(record_path, "w") as f:
        for off in offsetRanges:
            data = {"topic": off.topic, "partition": off.partition, \
                    "fromOffset": off.fromOffset, "untilOffset": off.untilOffset}
        f.write(json.dumps(data))

def assemble_kafka_setup(broker_list):
    """
    组装kafka的配置
    :param broker_list:
    :return:
    """
    kafkaParams = {"metadata.broker.list":broker_list}
    return kafkaParams

def deal_data(rdd):
    """
    处理数据模块
    """
    data = rdd.collect()
    sum = 0
    for d in data:
        sum += 1
        value = json.loads(d[-1])
        if value['_ch'] == 1:
            print('News : %s' % value['title'])
    print('Final deal data`sum is:',sum)

def kafka_direct(broker_list="",topic_list=[],query_time=10):
    """
    直连方式处理kafka数据
    :param broker_list:
    :param topic_list:
    :param query_time:
    :return:
    """
    root_path = os.path.dirname(os.path.realpath(__file__))
    record_path = os.path.join(root_path, OFFSETFILE)
    from_offsets = {}
    if os.path.exists(record_path):
        with open(record_path,'r') as f:
            offset_data = json.loads(f.read())
        topic_partion = TopicAndPartition(offset_data["topic"], offset_data["partition"])
        from_offsets = {topic_partion: int(offset_data["untilOffset"])}  # 注意设置起始offset时的方法
        print('start from offset:%s' % from_offsets)
    kafkaParams = assemble_kafka_setup(broker_list)
    sc = SparkContext(master="local[2]",appName="KafkaDirectApp")
    ssc = StreamingContext(sc, query_time)
    kvs = KafkaUtils.createDirectStream(ssc=ssc, kafkaParams=kafkaParams,topics=topic_list\
                                        ,fromOffsets=from_offsets)
    kvs.foreachRDD(lambda x:deal_data(x))
    kvs.transform(store_offset_ranges).foreachRDD(save_offset_ranges)
    ssc.start()
    ssc.awaitTermination()
    ssc.stop()
    sc.stop()

if __name__ == '__main__':
    kafka_direct('10.170.130.133:9092',['base_yt_news'])