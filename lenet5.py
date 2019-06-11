#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:28:29 2019

@author: Deechean
"""
import tensorflow as tf
import tf_general as tfg


class LeNet5(object):

    def __init__(self, x, n_class=10, keep_prob=1.0):
        self.input = x
        self.n_class = n_class
        self.keep_prob = keep_prob
        self._build_net()

    def _build_net(self):
        with tf.name_scope('norm'):    
            #self.x_norm = tf.nn.l2_normalize(tf.cast(self.input, tf.float32),dim=0)
            self.x_norm= tf.nn.lrn(self.input,depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75)
        
        with tf.name_scope('conv_1'):
            self.conv1 = tfg.conv2d(self.x_norm, 5, 1, 6, 'conv1', 'VALID','SIGMOID')
            print('conv_1: ', self.conv1.get_shape())
            
        with tf.name_scope('pool_1'):
            self.pool1 = tfg.avg_pool(self.conv1, 2, 2, 'pool1', 'VALID')
            print('pool_1: ', self.pool1.get_shape())
            
        with tf.name_scope('conv_2'):
            self.conv2 = tfg.conv2d(self.pool1, 5, 1, 16, 'conv2', 'VALID','SIGMOID')
            print('conv_2: ', self.conv2.get_shape())
            
        with tf.name_scope('pool_2'):
            self.pool2 = tfg.avg_pool(self.conv2, 2, 2, 'pool2', 'VALID')
            print('pool_2: ', self.pool2.get_shape())

        with tf.name_scope('flat_1'):
            self.flat1, self.flat_dim = tfg.flatten(self.pool2)
            print('flat_1:', self.flat1.get_shape())

        with tf.name_scope('fc_1'):
            self.fc1 = tfg.fc_layer(self.flat1, self.flat_dim, 120, 'fc1')
            print('fc_1: ', self.fc1.get_shape())

        with tf.name_scope('fc_2'):
            self.fc2 = tfg.fc_layer(self.fc1, 120, 84, 'fc2')
            print('fc_2: ', self.fc2.get_shape())
        
        with tf.name_scope('fc_3'):
            self.fc3 = tfg.fc_layer(self.fc2, 84, 10, 'fc3')
            print('fc_3: ', self.fc3.get_shape())

        with tf.name_scope('drop_out'):
            self.drop1 = tfg.drop_out(self.fc3, self.keep_prob, 'drop_out')
            print('drop_out: ', self.drop1.get_shape())

        with tf.name_scope('prediction'):
            self.prediction = tf.nn.softmax(self.drop1)
            print('prediction: ', self.prediction.get_shape())

class LeNet5_CIFAR10(object):

    def __init__(self, x, n_class=10, keep_prob=1.0):
        self.input = x
        self.n_class = n_class
        self.keep_prob = keep_prob
        self._build_net()

    def _build_net(self):
        with tf.name_scope('norm'):    
            #self.x_norm = tf.nn.l2_normalize(tf.cast(self.input, tf.float32),dim=0)
            self.x_norm= tf.nn.lrn(self.input,depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75)
        
        with tf.name_scope('conv_1'):
            self.conv1 = tfg.conv2d(self.x_norm, 5, 1, 12, 'conv1', 'VALID','SIGMOID')
            print('conv_1: ', self.conv1.get_shape())
            
        with tf.name_scope('pool_1'):
            self.pool1 = tfg.avg_pool(self.conv1, 2, 2, 'pool1', 'VALID')
            print('pool_1: ', self.pool1.get_shape())
            
        with tf.name_scope('conv_2'):
            self.conv2 = tfg.conv2d(self.pool1, 5, 1, 24, 'conv2', 'VALID','SIGMOID')
            print('conv_2: ', self.conv2.get_shape())
            
        with tf.name_scope('pool_2'):
            self.pool2 = tfg.max_pool(self.conv2, 2, 2, 'pool2', 'VALID')
            print('pool_2: ', self.pool2.get_shape())

        with tf.name_scope('flat_1'):
            self.flat1, self.flat_dim = tfg.flatten(self.pool2)
            print('flat_1:', self.flat1.get_shape())

        with tf.name_scope('fc_1'):
            self.fc1 = tfg.fc_layer(self.flat1, self.flat_dim, 120, 'fc1')
            print('fc_1: ', self.fc1.get_shape())

        with tf.name_scope('fc_2'):
            self.fc2 = tfg.fc_layer(self.fc1, 120, 84, 'fc2')
            print('fc_2: ', self.fc2.get_shape())
        
        with tf.name_scope('fc_3'):
            self.fc3 = tfg.fc_layer(self.fc2, 84, 10, 'fc3')
            print('fc_3: ', self.fc3.get_shape())

        with tf.name_scope('drop_out'):
            self.drop1 = tfg.drop_out(self.fc3, self.keep_prob, 'drop_out')
            print('drop_out: ', self.drop1.get_shape())

        with tf.name_scope('prediction'):
            self.prediction = tf.nn.softmax(self.drop1)
            print('prediction: ', self.prediction.get_shape()) 

            
class LeNet5_LERU(object):

    def __init__(self, x, n_class=10, keep_prob=1.0):
        self.input = x
        self.n_class = n_class
        self.keep_prob = keep_prob
        self._build_net()

    def _build_net(self):
        with tf.name_scope('norm'):    

            self.x_norm = tf.nn.l2_normalize(tf.cast(self.input, tf.float32),dim=0)
            #self.x_norm= tf.nn.lrn(self.input,depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75)
        
        with tf.name_scope('conv_1'):
            self.conv1 = tfg.conv2d(self.x_norm, 5, 1, 12, 'conv1', 'VALID')
            print('conv_1: ', self.conv1.get_shape())
            
        with tf.name_scope('pool_1'):
            self.pool1 = tfg.avg_pool(self.conv1, 2, 2, 'pool1', 'VALID')
            print('pool_1: ', self.pool1.get_shape())
            
        with tf.name_scope('conv_2'):
            self.conv2 = tfg.conv2d(self.pool1, 5, 1, 24, 'conv2', 'VALID')
            print('conv_2: ', self.conv2.get_shape())
            
        with tf.name_scope('pool_2'):
            self.pool2 = tfg.max_pool(self.conv2, 2, 2, 'pool2', 'VALID')
            print('pool_2: ', self.pool2.get_shape())

        with tf.name_scope('flat_1'):
            self.flat1, self.flat_dim = tfg.flatten(self.pool2)
            print('flat_1:', self.flat1.get_shape())

        with tf.name_scope('fc_1'):
            self.fc1 = tfg.fc_layer(self.flat1, self.flat_dim, 120, 'fc1')
            print('fc_1: ', self.fc1.get_shape())

        with tf.name_scope('fc_2'):
            self.fc2 = tfg.fc_layer(self.fc1, 120, 84, 'fc2')
            print('fc_2: ', self.fc2.get_shape())
        
        with tf.name_scope('fc_3'):
            self.fc3 = tfg.fc_layer(self.fc2, 84, 10, 'fc3')
            print('fc_3: ', self.fc3.get_shape())

        with tf.name_scope('drop_out'):
            self.drop1 = tfg.drop_out(self.fc3, self.keep_prob, 'drop_out')
            print('drop_out: ', self.drop1.get_shape())

        with tf.name_scope('prediction'):
            self.prediction = tf.nn.softmax(self.drop1)
            print('prediction: ', self.prediction.get_shape())

class LeNet5_LERU_ENH(object):

    def __init__(self, x, n_class=10, keep_prob=1.0):
        self.input = x
        self.n_class = n_class
        self.keep_prob = keep_prob
        self._build_net()

    def _build_net(self):
        with tf.name_scope('norm'):    
            self.x_norm = tf.nn.l2_normalize(tf.cast(self.input, tf.float32),dim=1)
            #self.x_norm= tf.nn.lrn(self.input,depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75)
        
        with tf.name_scope('conv_1'):
            self.conv1 = tfg.conv2d(self.x_norm, 5, 1, 18, 'conv1', 'VALID')
            print('conv_1: ', self.conv1.get_shape())
            
        with tf.name_scope('pool_1'):
            self.pool1 = tfg.avg_pool(self.conv1, 2, 2, 'pool1', 'VALID')
            print('pool_1: ', self.pool1.get_shape())
            
        with tf.name_scope('conv_2'):
            self.conv2 = tfg.conv2d(self.pool1, 5, 1, 48, 'conv2', 'VALID')
            print('conv_2: ', self.conv2.get_shape())
            
        with tf.name_scope('pool_2'):
            self.pool2 = tfg.max_pool(self.conv2, 2, 2, 'pool2', 'VALID')
            print('pool_2: ', self.pool2.get_shape())

        with tf.name_scope('flat_1'):
            self.flat1, self.flat_dim = tfg.flatten(self.pool2)
            print('flat_1:', self.flat1.get_shape())

        with tf.name_scope('fc_1'):
            self.fc1 = tfg.fc_layer(self.flat1, self.flat_dim, 120, 'fc1')
            print('fc_1: ', self.fc1.get_shape())

        with tf.name_scope('fc_2'):
            self.fc2 = tfg.fc_layer(self.fc1, 120, 84, 'fc2')
            print('fc_2: ', self.fc2.get_shape())
        
        with tf.name_scope('fc_3'):
            self.fc3 = tfg.fc_layer(self.fc2, 84, 10, 'fc3')
            print('fc_3: ', self.fc3.get_shape())

        with tf.name_scope('drop_out'):
            self.drop1 = tfg.drop_out(self.fc3, self.keep_prob, 'drop_out')
            print('drop_out: ', self.drop1.get_shape())

        with tf.name_scope('prediction'):
            self.prediction = tf.nn.softmax(self.drop1)
            print('prediction: ', self.prediction.get_shape())