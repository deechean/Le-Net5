#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:28:29 2019

@author: Deechean
"""
import tensorflow as tf
import tf_general as tfg


class LetNet5(object):

    def __init__(self, x, n_class=10, keep_prob=1.0):
        self.input = x
        self.n_class = n_class
        self.keep_prob = keep_prob
        self._build_net()

    def _build_net(self):
        with tf.name_scope('norm'):    
            self.x_norm = tf.nn.l2_normalize(tf.cast(self.input, tf.float32),dim=0)
        
        with tf.name_scope('conv_1'):
            self.conv1 = tfg.conv2d(self.x_norm, 5, 1, 6, 'conv1', 'VALID')
            
        with tf.name_scope('pool_1'):
            self.pool1 = tfg.avg_pool(self.conv1, 1, 1, 'pool1', 'VALID')
            
        with tf.name_scope('conv_2'):
            self.conv2 = tfg.conv2d(self.pool1, 5, 1, 16, 'conv2', 'VALID')
            
        with tf.name_scope('pool_2'):
            self.pool2 = tfg.avg_pool(self.conv2, 1, 1, 'pool2', 'VALID')
            
        with tf.name_scope('conv_3'):
            self.conv3 = tfg.conv2d(self.pool2, 5, 1, 120, 'conv3', 'VALID')

        with tf.name_scope('flat_1'):
            self.flat1, self.flat_dim = tfg.flatten(self.conv3)

        with tf.name_scope('fc_1'):
            self.fc1 = tfg.fc_layer(self.flat1, self.flat_dim, 84, 'fc1')

        with tf.name_scope('fc_2'):
            self.fc2 = tfg.fc_layer(self.fc1, 84, 10, 'fc2')

        with tf.name_scope('drop_out_1'):
            self.drop1 = tfg.drop_out(self.fc2, self.keep_prob, 'drop_out1')
            self.logits = self.drop1

        with tf.name_scope('prediction'):
            self.prediction = tf.nn.softmax(self.drop1)
            