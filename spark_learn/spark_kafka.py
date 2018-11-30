#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Kafka and spark test
Author LiHao
Time 2018/11/22 9:14
"""
import os
import sys
import logging
import json
import numpy as np
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils,TopicAndPartition
from ml_learn.algorithm.clusters import Dbscan
from golaxy_nlp.dataload import clean_sentence
from golaxy_nlp.similarity import _load_w2v
from golaxy_nlp.similarity import word2vec_transform,word2vec_transform_hierachical,word2vec_transform_maxpooling

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
TOPICNAME = 'news'
OFFSETFILE = TOPICNAME+'_offset'

LOG_DIR = 'logs'
LOG_FILENAME = TOPICNAME+'-kafka.log'
LOGFILE_REALPATH = os.path.join(os.path.join(ROOT_PATH,LOG_DIR),LOG_FILENAME)
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(processName)s - %(message)s'

logging.basicConfig(filename=LOGFILE_REALPATH,level=logging.INFO,format=LOG_FORMAT)
offsetRanges = []

dbscan = Dbscan(epos=2,minpts=2)
w2v = _load_w2v()

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
    record_path = os.path.join(ROOT_PATH, OFFSETFILE)
    data = dict()
    with open(record_path, "w") as f:
        for off in offsetRanges:
            data = {"topic": off.topic, "partition": off.partition, \
                    "fromOffset": off.fromOffset, "untilOffset": off.untilOffset}
            logging.warning("save offset : %s" % off.untilOffset)
        f.write(json.dumps(data))

def assemble_kafka_setup(broker_list):
    """
    组装kafka的配置
    :param broker_list:
    :return:
    """
    kafkaParams = {"metadata.broker.list":broker_list}
    return kafkaParams

def cluster(filename):
    word2vec_vectors = []
    for f in filename:
        cf = clean_sentence(f)
        wf = word2vec_transform(w2v,cf)
        word2vec_vectors.append(wf)
    v = np.array(word2vec_vectors)
    dbscan.fit(v)
    print("make %d clusters" %len(dbscan.group))

def deal_data(rdd):
    """
    处理数据模块
    """
    data = rdd.collect()
    sum = 0
    filename = []
    for d in data:
        sum += 1
        value = json.loads(d[-1])
        if value['_ch'] == 1:
            logging.info('News : %s' % value['title'])
            filename.append(value['title'])
    cluster(filename)
    logging.info('Final deal data`sum is: %s' % sum)

def kafka_direct(broker_list="",topic_list=[],query_time=10):
    """
    直连方式处理kafka数据
    :param broker_list:
    :param topic_list:
    :param query_time:
    :return:
    """
    record_path = os.path.join(ROOT_PATH, OFFSETFILE)
    from_offsets = {}
    if os.path.exists(record_path):
        with open(record_path,'r') as f:
            offset_data = json.loads(f.read())
        topic_partion = TopicAndPartition(offset_data["topic"], offset_data["partition"])
        from_offsets = {topic_partion: int(offset_data["untilOffset"])}  # 注意设置起始offset时的方法
        logging.info('start from offset:%s' % from_offsets)
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