#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:28:29 2019

@author: Deechean
"""
import sys
import os
import tensorflow as tf
sys.path.append('../common/')
import tf_general as tfg

class LeNet5(object):
    def __init__(self, x, n_class=10, drop_rate=0):
        self.input = x
        self.n_class = n_class
        self.drop_rate = drop_rate
        self._build_net()

    def _build_net(self):                          
        with tf.name_scope('norm_0'):    
            self.x_norm_0 = tf.nn.lrn(self.input,depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75, name='norm0')
            print('norm_0: ', self.x_norm_0.get_shape())
            
        with tf.name_scope('conv_1'):
            self.conv1 = tfg.conv2d(self.x_norm_0, 5, 1, 6, 'conv1', 'VALID','RELU')
            print('conv_1: ', self.conv1.get_shape())
        
        with tf.name_scope('pool_1'):
            self.pool1 = tfg.avg_pool(self.conv1, 2, 2, 'pool1', 'VALID')
            print('pool_1: ', self.pool1.get_shape())
        
        with tf.name_scope('norm_1'):    
            self.x_norm_1 = tf.nn.lrn(self.pool1,depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75, name='norm1')
            
        with tf.name_scope('conv_2'):
            self.conv2 = tfg.conv2d(self.x_norm_1, 5, 1, 16, 'conv2', 'VALID','RELU')
            print('conv_2: ', self.conv2.get_shape())
        
        with tf.name_scope('norm_2'):    
            self.x_norm_2 = tf.nn.lrn(self.conv2,depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75, name='norm2')
            
        with tf.name_scope('pool_2'):
            self.pool2 = tfg.avg_pool(self.x_norm_2, 2, 2, 'pool2', 'VALID')
            print('pool_2: ', self.pool2.get_shape()) 
            
        with tf.name_scope('conv_3'):
            self.conv3 = tfg.conv2d(self.pool2, 5, 1, 120, 'conv3', 'VALID','RELU')
            print('conv_3:', self.conv3.get_shape())
    
        with tf.name_scope('flat_1'):
            self.flat1, self.flat_dim = tfg.flatten(self.conv3)
            print('flat_1:', self.flat1.get_shape())
        
        with tf.name_scope('fc_2'):
            self.fc2 = tfg.fc_layer(self.flat1, 120, 84, 'fc2','RELU')
            print('fc_2 ', self.fc2.get_shape())
        
        with tf.name_scope('fc_3'):
            self.fc3 = tfg.fc_layer(self.fc2, 84, 10, 'fc4','RELU')
            print('fc_3: ', self.fc3.get_shape())

        with tf.name_scope('drop_out'):
            self.drop1 = tfg.drop_out(self.fc3, self.drop_rate, 'drop_out')
            print('drop_out: ', self.drop1.get_shape())
 
        with tf.name_scope('prediction'):
            self.prediction = tf.nn.softmax(self.drop1)
            print('prediction: ', self.prediction.get_shape())