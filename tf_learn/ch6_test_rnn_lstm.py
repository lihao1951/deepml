#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
预测 sin曲线
@author LiHao
@date 2019/3/14
"""
import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
HIDDEN_SIZE=30
NUM_LAYERS=2
TIMESTEPS=10
TRARNING_STEPS=10000
BATCH_SIZE=32
TRARNING_EXAMPLES=10000
TESTING_EXAMPLES=1000
SAMPLES_GAP=0.01

def generate_data(seq):
    X=[]
    y=[]
    for i in range(len(seq)-TIMESTEPS):
        X.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(X,y,is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
    outputs,_ = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
    output = outputs[:,-1,:]
    with tf.variable_scope("model"):
        fc_w = tf.get_variable(name='fc_w',shape=[HIDDEN_SIZE,1],dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc_b = tf.get_variable(name='fc_b', shape=[1], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
    predictions = tf.matmul(output,fc_w)+fc_b
    if not is_training:
        return predictions,None,None
    loss = tf.losses.mean_squared_error(labels=y,predictions=predictions)
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    return predictions,loss,train_op

def train(sess,train_X,train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_X,train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X,y = ds.make_one_shot_iterator().get_next()
    with tf.variable_scope('model'):
        predictions,loss,train_op = lstm_model(X,y,True)
    sess.run(tf.global_variables_initializer())
    for i in range(TRARNING_STEPS):
        _,l = sess.run([train_op,loss])
        if i%100==0:
            print("train stepL "+str(i)+" , loss:"+str(l))

def run_eval(sess,test_x,test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))
    ds = ds.batch(1)
    X,y = ds.make_one_shot_iterator().get_next()
    print(tf.global_variables())
    with tf.variable_scope("model",reuse=tf.AUTO_REUSE):
        prediction,_,_ = lstm_model(X,[0.0],False)
        predictions = []
        labels = []
        for i in range(TESTING_EXAMPLES):
            p,l = sess.run([prediction,y])
            predictions.append(p)
            labels.append(l)
        predictions = np.array(predictions).squeeze()
        labels = np.array(labels).squeeze()
        rmse = np.sqrt(((predictions-labels)**2).mean(axis=0))
        print("Mean Square Error is :%f" % rmse)
        plt.figure()
        plt.plot(predictions,label='predictions')
        plt.plot(labels,label='labels')
        plt.legend()
        plt.show()
test_start = (TRARNING_EXAMPLES+TRARNING_STEPS)*SAMPLES_GAP
test_end = test_start+(TESTING_EXAMPLES+TIMESTEPS)*SAMPLES_GAP
train_X,train_y = generate_data(np.sin(np.linspace(0,test_start,TRARNING_EXAMPLES+TIMESTEPS,dtype=np.float32)))
test_X,test_y = generate_data(np.sin(np.linspace(test_start,test_end,TESTING_EXAMPLES+TIMESTEPS,dtype=np.float32)))
with tf.Session() as sess:
    train(sess,train_X,train_y)
    run_eval(sess,test_X,test_y)