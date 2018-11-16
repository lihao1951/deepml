#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/10/23 16:04
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py

def sigmoid(Z):
    """
    sigmoid函数
    :param Z:
    :return:
    """
    A = 1/(1+np.exp(Z*-1))
    cache = Z
    return A,cache

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)
    return dZ

def relu(Z):
    """
    ReLU函数
    :param Z:
    :return:
    """
    A = np.maximum(0,Z)
    cache = Z
    return A,cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)
    return dZ

def initialize_parameters(nx,nh,ny):
    """
    初始化两层神经网络的权重
    nx,nh,ny分别为输入层、隐含层、输出层的神经元个数
    W1 输入到隐层的权重,(nh,nx)
    b1 隐层的偏置,(nh,1)
    W2 隐层到输出层的权重,(ny,nh)
    b2 输出层的偏置,(ny,1)
    """
    np.random.seed(1)
    W1 = np.random.randn(nh,nx) * 0.01
    b1 = np.zeros((nh, 1))
    W2 = np.random.randn(ny,nh) * 0.01
    b2 = np.zeros((ny, 1))
    assert (W1.shape == (nh,nx))
    assert (b1.shape == (nh,1))
    assert (W2.shape == (ny,nh))
    assert (b2.shape == (ny,1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def initialize_parameters_deep(layer_dims):
    """
    初始化 深层网络的权重及偏置
    :param layer_dims: 元组或列表 按顺序表示每层（包括输入层）的神经元大小
    :return:
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for i in range(1,L):
        parameters['W'+str(i)] = np.random.rand(layer_dims[i],layer_dims[i-1])*0.01
        parameters['b'+str(i)] = np.zeros((layer_dims[i],1))
        assert (parameters['W' + str(i)].shape == (layer_dims[i], layer_dims[i-1]))
        assert (parameters['b' + str(i)].shape == (layer_dims[i], 1))
    return parameters


def linear_forward(A,W,b):
    """
    前馈
    :param A:
    :param W:
    :param b:
    :return:
    """
    Z = np.dot(W,A) + b
    assert (Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)
    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
    """
    前馈激活函数
    :param A_prev:
    :param W:
    :param b:
    :param activation:
    :return:
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A,activation_cache = sigmoid(Z)
    elif activation == "relu":
        A,activation_cache = relu(Z)
    assert (A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)
    return A,cache

def L_model_forward(X,parameters):
    """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2 # //代表整数除法  /表示浮点数除法
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters['b'+str(l)],"relu")
        caches.append(cache)
    AL,cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],"sigmoid")
    assert (AL.shape == X.shape[1])
    caches.append(cache)
    return AL,caches


def compute_cost(AL,Y):
    """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = (1./m)*(-np.dot(Y,np.log(AL).T)-np.dot(1-Y,np.log(1-AL).T))
    cost = np.squeeze(cost) # 将[17],[[17]]等数据变为 17
    assert (cost.shape == ())
    return cost

def linear_backward(dZ,cache):
    """
    反向传播
    :param dZ:
    :param cache:
    :return:
    """
    A_prev,W,b = cache
    m = A_prev.shape[1]
    dW = (1./m)*np.dot(dZ,A_prev.T)
    db = (1./m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache # activation_cache 每一层的激活值

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

# GRADED FUNCTION: L_model_backward

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers 正向传播得到缓存值 长度为L
    m = AL.shape[1] #样本的数量
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL 确保L的形状与AL相同

    # Initializing the backpropagation
    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL)) #此处是计算dZ^[L]中的激活偏差的一部分

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    #对最后一层的sigmoid方向传播 求得各个参数的偏差值
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache,"relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))

