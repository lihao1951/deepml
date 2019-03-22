#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
@date 2019/3/22
"""
import tensorflow as tf
import numpy as np
from flask import Flask,request,render_template

from predict_news_hot.trans_text import get_word_list,read_vocab

app = Flask(__name__)
topK = 20
MODEL_NAME = './model/'


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
    X_input = graph.get_operation_by_name('X_input').outputs[0]
    y_input = graph.get_operation_by_name('y_input').outputs[0]
    y_pre = graph.get_operation_by_name("y_pre").outputs[0]
    print(graph.get_operations())
    return sess,X_input,y_input,y_pre


tf_session,X_input,y_input,y_pre = get_tf_session(MODEL_NAME)
vocab = read_vocab()

@app.route('/')
def demo():
    return render_template('index.html')


@app.route('/predict/rnn',methods=['GET','POST'])
def predict_rnn():
    cont = request.form.get('cont')
    word_list = get_word_list(vocab,cont)
    x = np.reshape(np.array(word_list,dtype=np.int32),newshape=(1,topK))
    print(x)
    predict = tf_session.run(y_pre,feed_dict={X_input:x,y_input:np.array([[1]],dtype=np.float32)})
    print(predict)
    return render_template('predict.html',predict=predict)

if __name__ == '__main__':
    app.run(debug=True)
