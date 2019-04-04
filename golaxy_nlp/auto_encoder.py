#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Name auto_encoder
@Description
    自编码器
@Author LiHao
@Date 2019/4/4
"""
import tensorflow as tf
from dl_learn.utils import split_toutiao_to_train_test
from dl_learn.utils import read_vocab

class AutoEncoder(object):
    def __init__(self,words_length,kernels,vocab_size,embedding_size,num_kernels,hidden_size):
        with tf.name_scope('input'):
            self.x = tf.placeholder(dtype=tf.int32,shape=[None,words_length],name='x')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        with tf.variable_scope('embedding'):
            self.embedding_weights = tf.get_variable(name='matrix',shape=[vocab_size,embedding_size],dtype=tf.float32
                                                     ,initializer=tf.truncated_normal_initializer(stddev=0.05))
            self.embedding_x = tf.nn.embedding_lookup(self.embedding_weights,self.x,name='x')
            self.embedding_x_expand = tf.expand_dims(self.embedding_x,name='expand_x',axis=-1)

        pooled_outputs = []
        for num, kernel in enumerate(kernels):
            with tf.name_scope('kernel_%s' % num):
                w = tf.Variable(tf.truncated_normal(shape=[kernel,embedding_size,1,num_kernels],dtype=tf.float32),name='w')
                b = tf.Variable(tf.constant(0.0,shape=[num_kernels],dtype=tf.float32),name='b')
                conv = tf.nn.conv2d(self.embedding_x_expand,w,strides=[1,1,1,1],padding='VALID',name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv,b))
                pool = tf.nn.max_pool(h,ksize=[1,words_length-kernel+1,1,1],strides=[1,1,1,1],padding='VALID',name='pool')
                pooled_outputs.append(pool)
        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, len(kernels)*num_kernels])

        with tf.name_scope('droupout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('hidden'):
            self.hidden_weights = tf.Variable(tf.truncated_normal(stddev=0.1,dtype=tf.float32,shape=[len(kernels)*num_kernels,hidden_size])
                                      ,name='weights')
            self.hidden_bias = tf.Variable(tf.constant(0.01,shape=[hidden_size],dtype=tf.float32),name='bias')
            self.hidden_layers = tf.nn.relu(tf.matmul(self.h_drop,self.hidden_weights)+self.hidden_bias,name='layers')

        with tf.name_scope('output'):
            self.out_weights = tf.Variable(tf.truncated_normal(stddev=0.1,shape=[hidden_size,words_length*embedding_size],dtype=tf.float32)
                                           ,name='weights')
            self.out_bias = tf.Variable(tf.constant(0.01,shape=[words_length*embedding_size],dtype=tf.float32),name='bias')

            self.out_layers = tf.nn.relu(tf.matmul(self.hidden_layers,self.out_weights)+self.out_bias,name='layers')

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.out_layers-tf.reshape(self.embedding_x,shape=[-1,words_length*embedding_size]))))

        with tf.name_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)



def main(batch_size = 32,train_epochs = 1000,dropout_prob=0.5):
    # 读取词典
    vocab = read_vocab()
    vocab_size  = len(vocab)
    train_x, train_y, test_x, test_y, label = split_toutiao_to_train_test(test_size=0.02)
    words_length = train_x.shape[-1]
    encoder = AutoEncoder(words_length=words_length, kernels=[3, 4, 5], vocab_size=vocab_size, embedding_size=100, num_kernels=128,
                          hidden_size=80)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1,1+train_epochs):
            # 每次迭代均生成批量数据   利用Dataset生成batch数据
            step_start = 0
            go_batch = True
            all_steps = 1
            while go_batch:
                x = train_x[step_start:(step_start+batch_size)]
                if x.shape[0] == 0:
                    print('---------next epoch---------')
                    go_batch = False
                else:
                    step_start += batch_size
                    _,train_loss = sess.run([encoder.train_op,encoder.loss],
                                                           feed_dict={encoder.x:x,encoder.dropout_keep_prob:dropout_prob})
                    print('epoch:{},steps:{},train loss:{}'.format(epoch, all_steps, train_loss))
                    all_steps+=1


if __name__ == '__main__':
    tf.app.run(main(64))