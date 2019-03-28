#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Date 2019/03/28
"""
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

def load_saver(model_name):
    # 读取最后一个模型的名称路径
    model_file_path = tf.train.latest_checkpoint(model_name)
    meta_file_path = model_file_path +'.meta'
    saver = tf.train.import_meta_graph(meta_file_path)
    return saver,model_file_path


def get_tf_session(model_name):
    saver,model_file_path = load_saver(model_name)
    sess = tf.Session()
    saver.restore(sess,model_file_path)
    graph = sess.graph
    X_input = graph.get_operation_by_name('input_x').outputs[0]
    y_input = graph.get_operation_by_name('input_y').outputs[0]
    y_pre = graph.get_operation_by_name("output/predictions").outputs[0]
    dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
    return sess,X_input,y_input,y_pre,dropout_keep_prob

sess,X_input,y_input,y_pre,dropout_keep_prob = get_tf_session('../model/toutiao/textcnn/')
from dl_learn.utils import split_toutiao_to_train_test
trainx,trainy,testx,testy,label=split_toutiao_to_train_test(0.1)
testy_pred = sess.run(y_pre,feed_dict={X_input:testx,y_input:testy,dropout_keep_prob:1.0})

def change(a):
    pred = []
    for i in range(a.shape[0]):
        pred.append(int(np.argmax(a[i])))
    return pred
a = change(testy)
b = testy_pred.tolist()
from dl_learn.rnn.lstm import text_labels
print(classification_report(a,b,target_names=[text_labels[num] for num in label]))



