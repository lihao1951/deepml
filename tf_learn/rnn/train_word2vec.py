#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Tensorflow实战
ch7 训练word2vec
Author LiHao
Time 2018/11/20 16:09
"""
import os
import sys
import collections
import math
import random
import zipfile
import numpy as np
from urllib import request
import tensorflow as tf

vocabulary_size = 50000
url = 'http://mattmahoney.net/dc/'
filename = 'text8.zip'

def maybe_download(filename,expected_bytes):
    """
    下载训练数据
    如果存在就返回错误，不存在就下载
    filename = maybe_download('text8.zip',31344016)
    :param filename:
    :param expected_bytes:
    :return:
    """
    if not os.path.exists(filename):
        filename,_= request.urlretrieve(url+filename,filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and Verfified:',filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to Verify' + filename + '. Can you get to it with a browser?')
    return filename

def read_data(filename):
    """
    读取文本内容
    :param filename:
    :return:
    """
    with zipfile.ZipFile(filename) as f:
        """
        type(f.read(f.namelist()[0])) == 'bytes'
        """
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)

def build_dataset(words):
    """
    生成词典、词典索引等
    del words
    print('Most common words:',count[:5])
    print('Sample data',data[:10],[reverse_dictionary[i] for i in data[:10]])

    :param words:
    :return:
    """
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = dict()
    for word , _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverse_dictionary

data,count,dictionary,reverse_dictionary = build_dataset(words=words)
del words
print('Most common words:',count[:5])
print('Sample data',data[:10],[reverse_dictionary[i] for i in data[:10]])

data_index = 0

def generate_batch(batch_size,num_skips,skip_windows):
    """
    生成数据batch训练序列
    主要就是生成训练数据批量对
    batch,labels = generate_batch(batch_size=8,num_skips=4,skip_windows=2)
    :param batch_size:
    :param num_skips:
    :param skip_windows:
    :return:
    """
    global data_index
    assert batch_size%num_skips ==0
    assert num_skips<=skip_windows*2
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span = 2 * skip_windows + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1)%len(data)
    for i in range(batch_size//num_skips):
        target = skip_windows
        targets_to_avoid = [skip_windows]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span-1)
            targets_to_avoid.append(target)
            batch[i*num_skips+j] = buffer[skip_windows]
            labels[i*num_skips+j,0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)
    return batch,labels

batch_size = 128
embedding_size = 128
skip_windows = 1
num_skips = 2

valid_size = 16 #验证单词数
valid_window = 100 #采样的范围
valid_examples = np.random.choice(valid_window,valid_size,replace=False)
num_sampled = 64 #负样本的个数
print(valid_examples)

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
    train_labels = tf.placeholder(tf.int32,shape=[batch_size,1])
    valid_dataset = tf.constant(valid_examples,dtype=tf.int32)
    with tf.device('/cpu:0'):
        # 创建嵌入矩阵
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
        # 嵌入层
        embed = tf.nn.embedding_lookup(embeddings,train_inputs)
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))
        nce_biaes = tf.Variable(tf.zeros([vocabulary_size]))
    # 定义损失函数
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biaes
                                         ,labels=train_labels
                                         ,inputs=embed
                                         ,num_sampled=num_sampled
                                         ,num_classes=vocabulary_size))
    # 定义优化器SGD
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

