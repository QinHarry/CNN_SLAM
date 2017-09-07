__author__ = "Hao Qin"
__email__ = "awww797877@gmail.com"

import tensorflow as tf
import numpy as np


def conv2d(input, out_channels=1, kernel_size=(1, 1), strides=[1, 1, 1, 1], padding='SAME', has_bias=True, batch_size=128,
           name='conv'):
    input_shape = input.get_shape().as_list()
    weights = tf.get_variable(shape=[kernel_size[0], kernel_size[1], input_shape[3], out_channels],
                              initializer=tf.truncated_normal_initializer(stddev=0.1), name=name + '_weights') #/tf.sqrt(tf.div(batch_size, 2.0))
    conv = tf.nn.conv2d(input, weights, strides=strides, padding=padding, name=name)
    if has_bias:
        b = tf.get_variable(shape=[out_channels], initializer=tf.constant_initializer(0.0), name=name + '_bias')
        tf.nn.bias_add(conv, b)

    return conv

def relu(input, name='relu'):
    act = tf.nn.relu(input, name=name)
    return act

def tanh(input, name='tanh'):
    act = tf.nn.tanh(input, name=name)
    return act

def dropout(input, keep=1.0, name='drop'):
    drop = tf.nn.dropout(input, keep)
    return drop

def batch_norm(input, decay=0.9, eps=1e-5, name='batch_norm'):
    with tf.variable_scope(name) as scope:
        batch_norm = tf.contrib.layers.batch_norm(input, decay=decay, epsilon=eps, center=True, scale=True,
                                     is_training=1, reuse=None,
                                     updates_collections=None, scope=scope)

    return batch_norm

def flatten(input, name='flat'):
    input_shape = input.get_shape().as_list()        # list: [None, 9, 2]
    dim   = np.prod(input_shape[1:])                 # dim = prod(9,2) = 18
    flat  = tf.reshape(input, [-1, dim], name=name)  # -1 means "all"
    return flat


def maxpool(input, kernel_size = (1,1), strides = [1,1,1,1], padding = 'SAME', name = 'max' ):
    H = kernel_size[0]
    W = kernel_size[1]
    pool = tf.nn.max_pool(input, ksize = [1, H, W, 1], strides = strides, padding = padding, name = name)
    return pool

def average_pool(input, kernel_size=(1,1), strides=[1,1,1,1], padding='SAME', name='average'):
    H = kernel_size[0]
    W = kernel_size[1]
    pool = tf.nn.avg_pool(input, ksize = [1, H, W, 1], strides = strides, padding = padding, name = name)
    return pool

def conv2d_batch_norm_relu(input, out_channels, kernel_size, strides=[1,1,1,1], padding='SAME', has_bias=True, batch_size=128, name='conv'):
    with tf.variable_scope(name) as scope:
        block = conv2d(input, out_channels, kernel_size, strides, padding, has_bias, batch_size=batch_size)
        block = batch_norm(block)
        block = relu(block)
    return block

def l2_regulariser(decay):

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in variables:
        name = v.name
        if 'weight' in name:  #this is weight
            l2 = decay * tf.nn.l2_loss(v)
            tf.add_to_collection('losses', l2)
        elif 'bias' in name:  #this is bias
            pass
        elif 'beta' in name:
            pass
        elif 'gamma' in name:
            pass
        elif 'moving_mean' in name:
            pass
        elif 'moving_variance' in name:
            pass
        elif 'moments' in name:
            pass

        else:
            #pass
            raise Exception('unknown variable type: %s ?'%name)
            pass

    l2_loss = tf.add_n(tf.get_collection('losses'))
    return l2_loss

def linear(input, num_hiddens=1,  has_bias=True, name='linear'):
    input_shape = input.get_shape().as_list()
    assert len(input_shape)==2

    C = input_shape[1]
    K = num_hiddens

    w = tf.get_variable(name=name + '_weight', shape=[C,K], initializer=tf.truncated_normal_initializer(stddev=0.1))
    dense = tf.matmul(input, w, name=name)
    if has_bias:
        b = tf.get_variable(name=name + '_bias', shape=[K], initializer=tf.constant_initializer(0.0))
        dense = dense + b

    return dense

def linear_batch_norm_relu(input, num_hiddens=1, name='conv'):
    with tf.variable_scope(name) as scope:
        block = linear(input, num_hiddens=num_hiddens, has_bias=False)
        block = batch_norm(block, num_hiddens)
        block = relu(block)
    return block

def linear_relu(input, num_hiddens=1, name='linear'):
    with tf.variable_scope(name) as scope:
        block = linear(input, num_hiddens=num_hiddens, has_bias=False)
        block = relu(block)
    return block
