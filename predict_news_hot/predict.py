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
BATCH_SIZE = 100
HIDDEN_SIZE = 80
NUM_LAYERS = 2
EMBEDDING_OUT_PROB = 0.7
LSTM_OUT_PROB = 0.7
TRAIN_STEPS = 1000
CNN_TENSORBOARD_PATH = './tensorboard/predict/cnn/'
RNN_TENSORBOARD_PATH = './tensorboard/predict/cnn/'
CNN_MODEL_PATH = './model/cnn/cnn_predict.ckpt'
RNN_MODEL_PATH = './model/rnn/rnn_predict.ckpt'
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


def cnn_weight(shape,name=None):
    initial = tf.truncated_normal(shape=shape,stddev=0.1,name = name,dtype=tf.float32)
    return tf.Variable(initial_value=initial)


def cnn_bias(shape,name=None):
    initial = tf.constant(value=0.1,shape=shape,dtype=tf.float32)
    return tf.Variable(initial)


# 读取词库
VOCAB_DICT = read_vocab()


def cnn_get_loss_train_op():
    g = tf.Graph()
    with g.as_default():
        X_input = tf.placeholder(dtype=tf.int32, shape=[None, NUM_STEPS], name='X_input')
        y_input = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='y_input')
        y_input_one_hot = tf.reshape(tf.one_hot(y_input,depth=2,dtype=tf.float32,axis=1),shape=[-1,2],name='y_input_one_hot')
        embedding = tf.get_variable(name='embedding_layer', shape=[len(VOCAB_DICT), HIDDEN_SIZE],
                                    initializer=tf.random_uniform_initializer(minval=-1.0,maxval=1.0),
                                    dtype=tf.float32)
        tf.summary.histogram('embedding',embedding)
        X_embedding = tf.reshape(tf.nn.embedding_lookup(embedding,X_input),shape=[-1,NUM_STEPS,HIDDEN_SIZE,1]
                                 ,name='X_embedding')
        # 第一卷积层+最大池化层
        conv1_filter = cnn_weight([3,5,1,32],name='conv1_filter')
        conv1_bias = cnn_bias([32],name='conv1_bias')
        conv1 = tf.nn.bias_add(tf.nn.conv2d(input=X_embedding,filter=conv1_filter,strides=[1,1,1,1],padding='SAME')
                               ,conv1_bias,name='conv1')
        h_conv1= tf.nn.relu(conv1,name='h_conv1')
        tf.summary.histogram('h_conv1', h_conv1)
        pool1 = tf.nn.max_pool(h_conv1,ksize=[1,1,2,1],strides=[1,1,2,1],name='pool1',padding='VALID')
        # 第二卷积层+最大池化层
        conv2_filter = cnn_weight([3,5,32,64],name='conv2_filter')
        conv2_bias = cnn_bias([64], name='conv2_bias')
        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1,filter=conv2_filter,padding='SAME',strides=[1,1,1,1]),
                               conv2_bias,name='conv2')
        h_conv2 = tf.nn.relu(conv2,name='h_conv2')
        tf.summary.histogram('h_conv2', h_conv2)
        pool2 = tf.nn.max_pool(h_conv2,ksize=[1,1,2,1],strides=[1,1,2,1],name='pool2',padding='VALID')
        # 第三卷积层+最大池化层
        conv3_filter = cnn_weight([5,5,64,128],name='conv3_filter')
        conv3_bias = cnn_bias([128],name='conv3_bias')
        conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2,filter=conv3_filter,strides=[1,1,1,1],padding='SAME'),conv3_bias
                               ,name='conv3')
        h_conv3 = tf.nn.relu(conv3,name='h_conv3')
        tf.summary.histogram('h_conv3', h_conv3)
        pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],name='pool3',padding='VALID')
        # 第四卷积层+最大池化层
        conv4_filter = cnn_weight([3,3,128,128],name='conv4_filter')
        conv4_bias = cnn_bias([128],name='conv4_bias')
        conv4 = tf.nn.bias_add(tf.nn.conv2d(pool3,filter=conv4_filter,strides=[1,1,1,1],padding='SAME'),conv4_bias,
                               name='conv4')
        h_conv4 = tf.nn.relu(conv4,name='h_conv4')
        tf.summary.histogram('h_conv4', h_conv4)
        pool4 = tf.nn.max_pool(h_conv4,ksize=[1,2,2,1],strides=[1,2,2,1],name='pool4',padding='VALID')

        conv_input = tf.reshape(pool4,shape=[-1,5*5*128],name='conv_input')

        w_fc1 = cnn_weight([5*5*128,1024],name='w_fc1')
        b_fc1 = cnn_bias([1024],name='b_fc1')
        w_fc2 = cnn_weight([1024,256],name='w_fc2')
        b_fc2 = cnn_bias([256],name='b_fc2')
        w_fc3 = cnn_weight([256,32],name='w_fc3')
        b_fc3 = cnn_bias([32],name='b_fc3')
        w_fc4 = cnn_weight([32,2],name='w_fc4')
        b_fc4 = cnn_bias([2],name='b_fc4')

        h_fc1 = tf.nn.relu(tf.matmul(conv_input,w_fc1)+b_fc1,name='h_fc1')
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1,w_fc2)+b_fc2,name='h_fc2')
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3,name='h_fc3')
        y_pre = tf.nn.softmax(tf.matmul(h_fc3,w_fc4)+b_fc4,name='y_pre')
        tf.summary.histogram('h_fc1', h_fc1)
        tf.summary.histogram('h_fc2', h_fc2)
        tf.summary.histogram('h_fc3', h_fc3)
        loss = tf.reduce_mean(-tf.reduce_sum(y_input_one_hot*tf.log(y_pre)))
        tf.summary.scalar('loss',loss)
        correct_value = tf.equal(tf.argmax(y_input_one_hot,1),tf.argmax(y_pre,1))
        accuracy = tf.reduce_mean(tf.cast(correct_value,dtype=tf.float32),name='accuracy')
        tf.summary.scalar('accuracy',accuracy)
        merged = tf.summary.merge_all()
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

        return g,train_op,loss,accuracy,X_input,y_input,merged


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
            y_pre = tf.nn.sigmoid(tf.add(tf.matmul(final_output, fc_w), fc_b),name='y_pre')
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
        writer = tf.summary.FileWriter(RNN_TENSORBOARD_PATH, g)
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
                    break
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
            saver.save(sess, save_path=RNN_MODEL_PATH, global_step=i + 1)


def compute_y(y):
    value_map = {0.0:0,1.0:0}
    for row in range(y.shape[0]):
        if y[row,0]==0.0:
            value_map[0.0] += 1
        else:
            value_map[1.0] += 1
    sum = value_map[0.0] + value_map[1.0]
    return 'num of 0 :{} / num of 1:{} / sum :{}'.format(value_map[0.0],value_map[1.0],sum)


def cnn_train():
    g,train_op,loss,accuracy,X_input,y_input,merged = cnn_get_loss_train_op()
    writer = tf.summary.FileWriter(CNN_TENSORBOARD_PATH,graph=g)
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=2)
        all_steps = 0
        for epoch in range(1,TRAIN_STEPS+1):
            go_on = True
            iters = 0
            train_data = get_batch_data(TRAIN_PATH, batch_size=BATCH_SIZE)
            all_train_loss = 0
            all_train_accuracy = 0
            while go_on:
                try:
                    train_x,train_y = next(train_data)
                    iters += 1
                    all_steps += 1
                    _,train_loss,train_accuracy = sess.run([train_op,loss,accuracy],feed_dict={X_input:train_x,y_input:train_y})
                    all_train_loss += train_loss
                    all_train_accuracy += train_accuracy
                    if iters % 10 == 0:
                        rs = sess.run( merged, feed_dict={X_input: train_x, y_input: train_y})
                        writer.add_summary(rs, global_step=all_steps)
                        print('epoch:{},iter:{},train loss:{},train accuracy:{}'.format(epoch,iters,train_loss,train_accuracy))
                except Exception as e:
                    go_on = False
                    print(e)
            valid_data = get_batch_data(VALID_PATH, batch_size=1000)
            valid_x, valid_y = next(valid_data)
            valid_accuracy = run_eval(sess, accuracy, feed={X_input: valid_x, y_input: valid_y})
            print(compute_y(valid_y))
            print('epoch:{},all train loss:{},all train accuracy:{},valid loss:{}'.format(epoch, all_train_loss / iters,
                                                                                          all_train_accuracy / iters,
                                                                                          valid_accuracy))
            saver.save(sess, save_path=CNN_MODEL_PATH, global_step=epoch)


def main(model):
    if model is 'rnn':
        rnn_train(False)
    elif model is 'cnn':
        cnn_train()
    else:
        raise NotImplementedError("没找到对应的模型")

if __name__ == '__main__':
    tf.app.run(main('cnn'))