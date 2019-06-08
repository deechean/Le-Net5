#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:37:08 2019

@author: Deechean
"""

import tensorflow as tf
import os 
import matplotlib.pyplot as plt
import numpy as np
import time

def get_variable(name, shape, initializer, regularizer=None, dtype='float', trainable=True):
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           regularizer=regularizer,
                           collections=collections,
                           dtype=dtype,
                           trainable=trainable)
    #tf.get_variable_scope().reuse_variables()

def conv2d(x, ksize, stride, filter_out, name, padding='VALID', activate = 'RELU'):
    """ 
    x: input 
    ksize: kernel size 
    stride
    filter_out: filters numbers
    name: name of the calculation
    padding: VALID - no padding, SAME - keep the output size same as input size
    activate: RELU - relu or SIGMOID  -sigmoid
    """
    with tf.variable_scope(name):
        #Get input dimention
        filter_in = x.get_shape()[-1]
        
        stddev = 1. / tf.sqrt(tf.cast(filter_out, tf.float32))
        #use random uniform to initialize weight
        weight_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)
        #use random uniform to initialize bias
        bias_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)
        #kernel shape is [kenel size, kernel size, filter in size, filter out size]
        shape = [ksize, ksize, filter_in, filter_out]
        #set kernel
        kernel = get_variable('kernel', shape, weight_initializer)
        #set bias, bias shape is [filter_out]
        bias = get_variable('bias', [filter_out], bias_initializer)
        #conv2d
        conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding=padding)
        #add conv result with bias
        out = tf.nn.bias_add(conv, bias)
        #activate
        if activate == 'RELU':
            out = tf.nn.relu(out)
        elif activate == 'SIGMOID':
            out = tf.nn.sigmoid(out)
        
        return out
    
    
def max_pool(x, ksize, stride, name, padding):
    """ x: input
        ksize: kernel size
        stride: stride
        name: name of the calculation
        padding: VALID - no padding, SAME - keep the output size same as input size
    """
    return tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], name=name, padding=padding)

def avg_pool(x, ksize, stride, name, padding):
    """ average pool
        x: input
        ksize: kernel size
        stride: stride
        name: name of the calculation
        padding: VALID - no padding, SAME - keep the output size same as input size
    """    
    return tf.nn.avg_pool(x, [1, ksize, ksize, 1],[1, stride, stride, 1], name=name, padding=padding)

def flatten(x):
    """Reshape x to a list(one dimesion)
    """    
    shape = x.get_shape().as_list()
    dim = 1
    for i in range(1, len(shape)):
        dim *= shape[i]
    return tf.reshape(x, [-1, dim]), dim

def fc_layer(x, i_size, o_size, name, is_relu=None):
    """Full connection layer
        x:
        i_size: input size
        o_size: output size
        name: name of the calculation
        is_relu: 
    """
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape=[i_size, o_size], dtype='float')
        b = tf.get_variable('b', shape=[o_size], dtype='float')
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if is_relu:
            out = tf.nn.relu(out)
        return out

def drop_out(x, keep_prob, name):
    """drop out to prevent overfit, it should only used in training, not in test
        x: input
        keep_prob: probability of drop out, normally is 0.5
        name: name of the calculation
        
    """
    return tf.nn.dropout(x, keep_prob=keep_prob, name=name)
    
def saveEvalData(file,datalist):
    with open(os.getcwd()+"/"+file,'a+',encoding='utf-8') as f:
        for x in datalist:
            f.write(str(x) + '\n')

def isIter(variable):
    try:
        iter(variable)
        return True
    except:
        return False
    
def strlize(variable):   
    if isIter(variable):
        valuestr = '['
        for item in variable:
            valuestr += strlize(item) + ','
        valuestr = valuestr[:len(valuestr)-1]+']'
    else:
        valuestr = str(variable)
    return valuestr
      
def savelog(logdir, variable, globalstep, value):
    #filename = 'lenet_train_cifar10.log'
    valuestr = strlize(value)        
    log = []
    log.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+ ', global step: '+ str(globalstep) + ', '+ variable +':'+valuestr)
    saveEvalData(logdir+variable, log)

def readlog(logfile, variable):
    parameterlist = []
    with open(logfile,'r',encoding='utf-8') as f:
        while True: 
            line = f.readline()
            if line == '':
                break
            if line.find(' '+variable+':') > 0:
                pos = line.find('global step:')
                if pos > 0: 
                    line = line[line.find('global step:')+len('global step:'):]
                    step  = int(line[:line.find(',')])
                    value = line[line.find(variable+':')+len(variable+':'):line.find('\n')]
                    try:
                        value = float(value)
                    except ValueError:
                        if value[0] == '[':
                            value = value[1:]
                            itemlist = []
                            while value.find(",") > 0: 
                                item = float(value[:value.find(",")])
                                itemlist.append(item)
                                value = value[value.find(",")+1:]
                            item = float(value[:-1])
                            itemlist.append(item)
                        value = itemlist                                    
                parameterlist.append([step, value]) 
                #print([step, value])
    return parameterlist
    
def printimages(images):
    for img in images:    
        plt.imshow(np.asarray(img).reshape(32,32,3))
        plt.show()