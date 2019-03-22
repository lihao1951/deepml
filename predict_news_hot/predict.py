#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author LiHao
@date 2019/3/18
"""
import tensorflow as tf
import codecs
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn

TRAIN_PATH = 'mini_train_sig'
VALID_PATH = 'mini_valid_sig'
TEST_PATH = 'mini_test_sig'
NUM_STEPS = 20
BATCH_SIZE = 64
HIDDEN_SIZE = 200
NUM_LAYERS = 2
EMBEDDING_OUT_PROB = 0.7
LSTM_OUT_PROB = 0.7
TRAIN_STEPS = 100
TENSORBOARD_PATH = './tensorboard/'
MODEL_PATH = './model/predict.ckpt'
PRINT_TRAIN_LOG = True


def read_vocab():
    vocab = {}
    with open('vocab', 'r', encoding='utf-8') as fin:
        for num,word in enumerate(fin.readlines()):
            vocab[word.strip()] = num
        vocab['<pad>'] = len(vocab)
        vocab['<eos>'] = len(vocab)
        vocab['<sos>'] = len(vocab)
    return vocab


def get_batch_data(filename,batch_size=2):
    fin = codecs.open(filename,'r',encoding='utf-8')
    line = fin.readline()
    datas = []
    labels = []
    while line:
        meta_data = line.strip().split('-')
        if len(datas) >= batch_size:
            yield np.array(datas,np.int32),np.array(labels,np.float32)
            datas = []
            labels = []
        datas.append(meta_data[1].split())
        labels.append([float(meta_data[0])])
        line = fin.readline()
    yield np.array(datas,np.int32),np.array(labels,np.float32)


# 读取词库
VOCAB_DICT = read_vocab()


def rnn_get_loss_train_op(if_pre = True):
    # 定义一个新的graph
    g = tf.Graph()
    with g.as_default():
        X_input = tf.placeholder(dtype=tf.int32, shape=[None, NUM_STEPS], name='X_input')
        y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')
        embedding = tf.get_variable(name='embedding', shape=[len(VOCAB_DICT), HIDDEN_SIZE], dtype=tf.float32)
        # embedding_input的shape is [batch_size,num_steps,hidden_size]
        tf.summary.histogram('embedding', embedding)
        embedding_input = tf.nn.embedding_lookup(embedding, X_input, name='embedding_input')
        lstm_units = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE), output_keep_prob=LSTM_OUT_PROB)
            for _ in range(NUM_LAYERS)]
        rnncell = tf.nn.rnn_cell.MultiRNNCell(lstm_units)
        outputs, state = dynamic_rnn(rnncell, embedding_input, dtype=tf.float32)
        # 创建
        final_output = outputs[:, -1, :]
        # 形成全连接FC层
        with tf.variable_scope('FC', dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1)):
            fc_w = tf.get_variable('w', shape=[HIDDEN_SIZE, 1])
            fc_b = tf.get_variable('b', shape=[1])
            tf.summary.histogram('FC_w', fc_w)
            tf.summary.histogram('FC_b', fc_b)
        if if_pre:
            y_pre = tf.add(tf.matmul(final_output, fc_w), fc_b)
            loss = tf.sqrt(tf.losses.mean_squared_error(labels=y_input, predictions=y_pre))
            tf.summary.scalar('loss', loss)
        else:
            y_pre = tf.nn.sigmoid(tf.add(tf.matmul(final_output, fc_w), fc_b))
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input,logits=y_pre))
            one = tf.ones_like(y_pre)
            zero = tf.zeros_like(y_pre)
            accuracy = 1.0 - tf.to_float(tf.reduce_mean(tf.abs(tf.cast(tf.where(y_pre>=0.6,one,zero),dtype=tf.float32)-y_input)))
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('loss', loss)
        # 定义损失函数及训练方法
        # tf.summary.scalar('loss', loss)
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        merged = tf.summary.merge_all()
        if if_pre:
            return g,loss,train_op,X_input,y_input,merged
        else:
            return g,loss,train_op,X_input,y_input,merged,accuracy


def run_eval(sess,op=None,feed={}):
    loss = sess.run(op,feed)
    return loss


def rnn_train(if_pre):
    if if_pre:
        # 预测任务
        g, loss, train_op, X_input, y_input, merged = rnn_get_loss_train_op(if_pre)
    else:
        # 分类任务
        g, loss, train_op, X_input, y_input, merged,accuracy = rnn_get_loss_train_op(if_pre)
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=2)  # 最多保留三个模型
        writer = tf.summary.FileWriter(TENSORBOARD_PATH, g)
        # 生成迭代数据
        all_steps = 0
        for i in range(TRAIN_STEPS):
            train_loss_sum = 0.0
            step = 0
            go_on = True
            train_data = get_batch_data(TRAIN_PATH, batch_size=BATCH_SIZE)
            while go_on:
                try:
                    train_X, train_y = next(train_data)
                    step += 1
                    all_steps += 1
                    _, train_loss,rs = sess.run([train_op, loss,merged], feed_dict={X_input: train_X, y_input: train_y})
                    if all_steps % 10 == 0:
                        writer.add_summary(rs, all_steps)
                    if (step % 10 == 0) and PRINT_TRAIN_LOG:
                        if if_pre:
                            print("epoch:{},iteration:{},train loss:{}".format(i + 1, step, train_loss))
                        else:
                            train_accuracy = sess.run(accuracy, feed_dict={X_input: train_X, y_input: train_y})
                            print("epoch:{},iteration:{},train loss:{},train accuracy:{}".format(i + 1, step, train_loss,train_accuracy))
                    train_loss_sum += train_loss
                except Exception as e:
                    go_on = False
                    print(e)
            valid_data = get_batch_data(VALID_PATH, batch_size=1000)
            valid_X, valid_y = next(valid_data)
            if if_pre:
                valid_loss = run_eval(sess,loss,feed={X_input: valid_X, y_input: valid_y})
                print('\t epoch:{},mean train loss:{},valid loss:{}'.format(i + 1, train_loss_sum / step, valid_loss))
            else:
                valid_loss,valid_accuracy = run_eval(sess,[loss,accuracy],feed={X_input: valid_X, y_input: valid_y})
                print('\t epoch:{},mean train loss:{},valid loss:{},valid accuracy:{}'.format(i + 1, train_loss_sum / step, valid_loss,valid_accuracy))
            saver.save(sess, save_path=MODEL_PATH, global_step=i + 1)


def cnn_train():
    raise NotImplementedError("该模型还没实现")


def main(model):
    if model is 'rnn':
        rnn_train(False)
    elif model is 'cnn':
        cnn_train()
    else:
        raise NotImplementedError("没找到对应的模型")

if __name__ == '__main__':
    tf.app.run(main('rnn'))