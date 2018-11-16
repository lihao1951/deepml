import numpy as np
import h5py
"""
此处将h5py升级到了2.8.0 可以运行成功
"""

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


from ml_learn.andrew_ng_course.C1.lr_utils import load_dataset


def deal_with_train_test_data():
    """
    读入数据
    （1）读入图像训练集及测试集大小为（64,64,3）
    （2）每一幅图像拉成（64*64*3）的列向量，并归一化 像素值/255
    （3）返回训练、测试数据、类别名称
    :return: X_train,y_train,X_test,y_test,classes
    """
    # %matplotlib inline此处设置为将图像嵌入到notebook中
    # 读取数据入库
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    # print("Number of training examples: m_train = " + str(m_train))
    # print("Number of testing examples: m_test = " + str(m_test))
    # print("Height/Width of each image: num_px = " + str(num_px))
    # print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    # print("train_set_x shape: " + str(train_set_x_orig.shape))
    # print("train_set_y shape: " + str(train_set_y.shape))
    # print("test_set_x shape: " + str(test_set_x_orig.shape))
    # print("test_set_y shape: " + str(test_set_y.shape))
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.
    return train_set_x,train_set_y,test_set_x,test_set_y,classes