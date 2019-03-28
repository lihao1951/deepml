#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
@date 2019/3/26
"""
import tensorflow as tf

from dl_learn.utils import split_toutiao_to_train_test
from dl_learn.utils import read_vocab

class TextCNN(object):
    """
    TextCNN model
    """
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,filter_sizes,num_filters,l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32,shape=[None,sequence_length],name='input_x')
        self.input_y = tf.placeholder(tf.float32,shape=[None,num_classes],name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

        l2_loss = tf.constant(0.0)

        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0,dtype=tf.float32),name='W')
            self.embedded_chars = tf.nn.embedding_lookup(self.W,self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars,-1)
        pooled_outputs = []
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s'% filter_size):
                filter_shape = [filter_size,embedding_size,1,num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1,dtype=tf.float32),name='W')
                b = tf.Variable(tf.constant(0.1,shape=[num_filters],dtype=tf.float32),name='b')
                conv = tf.nn.conv2d(self.embedded_chars_expanded,W,strides=[1,1,1,1],padding="VALID",name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
                pooled = tf.nn.max_pool(h,ksize=[1,sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name='pool')
                pooled_outputs.append(pooled)
        num_filters_total  = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

        with tf.name_scope('droupout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,keep_prob=self.dropout_keep_prob)

        with tf.name_scope('output'):
            W = tf.get_variable("W",shape=[num_filters_total,num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1,shape=[num_classes],name='b'))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop,W,b,name='scores')
            self.predictions = tf.argmax(self.scores,1,name='predictions')

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores,labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"),name='accuracy')
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)


def main(batch_size = 32,train_epochs = 1000,dropout_prob=0.5,save_path='../model/toutiao/textcnn/cnn.ckpt'):
    # 读取词典
    vocab = read_vocab()
    vocab_size  = len(vocab)
    train_x, train_y, test_x, test_y, label = split_toutiao_to_train_test(test_size=0.02)
    num_classes = len(label)
    train_rows = train_x.shape[0]
    sequence_length = train_x.shape[-1]

    # 构建TextCNN模型
    textcnn = TextCNN(sequence_length=sequence_length,num_classes=num_classes,vocab_size=vocab_size
                      ,embedding_size=64,filter_sizes=[3,4,5],num_filters=16,l2_reg_lambda=0.01)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
        for epoch in range(1,1+train_epochs):
            # 每次迭代均生成批量数据   利用Dataset生成batch数据
            if epoch>1:
                train_x, train_y, test_x, test_y, label = split_toutiao_to_train_test(test_size=0.02)
            step_start = 0
            go_batch = True
            all_steps = 1
            while go_batch:
                x = train_x[step_start:step_start+batch_size]
                y = train_y[step_start:step_start+batch_size]
                if x.shape[0] == 0:
                    print('---------next epoch---------')
                    go_batch = False
                else:
                    step_start += batch_size
                    _,train_loss,train_accuracy = sess.run([textcnn.train_op,textcnn.loss,textcnn.accuracy],
                                                           feed_dict={textcnn.input_x:x,textcnn.input_y:y,textcnn.dropout_keep_prob:dropout_prob})
                    if all_steps % 10 == 0:
                        test_loss, test_accuracy = sess.run([textcnn.loss, textcnn.accuracy],
                                                                 feed_dict={textcnn.input_x: test_x, textcnn.input_y: test_y,
                                                                            textcnn.dropout_keep_prob: 1.0})
                        print('epoch:{},steps:{},train loss:{},train accuracy:{},test loss:{},test accuracy:{}'.format(epoch, all_steps, train_loss,
                                                                                                                     train_accuracy, test_loss,test_accuracy))
                    all_steps+=1
            saver.save(sess,save_path=save_path,global_step=epoch)

if __name__ == '__main__':
    tf.app.run(main(batch_size=128))