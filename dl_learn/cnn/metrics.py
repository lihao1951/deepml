#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Date 2019/03/28
"""
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from predict_news_hot.trans_text import get_word_list,read_vocab

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


def change(a):
    pred = []
    for i in range(a.shape[0]):
        pred.append(int(np.argmax(a[i])))
    return pred


def classify(cont_list):
    vocab = read_vocab()
    from dl_learn.rnn.lstm import news_label_list, news_labels_name_dict
    sess,X_input,y_input,y_pre,dropout_keep_prob = get_tf_session('../model/toutiao/textcnn/')
    in_x = []
    in_y = []
    for cont in cont_list:
        wordlist = [int(word) for word in get_word_list(vocab,cont,topK=15)]
        in_x.append(wordlist)
        in_y.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    testy_pred = sess.run(y_pre,feed_dict={X_input:np.array(in_x,dtype=np.int32),y_input:np.array(in_y,dtype=np.float32),dropout_keep_prob:1.0})
    pred = np.squeeze(testy_pred).tolist()
    for num,x in enumerate(pred):print(num+1,'\t',news_labels_name_dict[news_label_list[x]])


if __name__ == '__main__':
    f = open('./test','r',encoding='utf-8')
    cont_list = [line.strip() for line in f.readlines()]
    classify(cont_list)