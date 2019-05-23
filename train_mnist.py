#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:52:29 2019

@author: Deechean
"""

import os
import tensorflow as tf
from lenet5 import LetNet5
from tensorflow.examples.tutorials.mnist import input_data
import tf_general as tfg
import numpy as np

FLAGS = tf.flags.FLAGS
try:
    tf.flags.DEFINE_integer('epoch', 100, 'epoch')
    tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
    tf.flags.DEFINE_integer('test_size', 10000, 'test size')
    tf.flags.DEFINE_float('lr', 0.001, 'learning rate')
    tf.flags.DEFINE_boolean('restore', False, 'restore from checkpoint and run test')
except:
    print('parameters have been defined.')

data_dir = 'mnist_data'
ckpt_dir = 'ckpt/'

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    y_ = tf.placeholder(tf.float32, [None, 10], name='label_input')

with tf.name_scope('prediction'):
    le_net5 = LetNet5(x_image, keep_prob)
    fc2 = le_net5.fc2
    logit = le_net5.logits
    y = le_net5.prediction

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

    
with tf.name_scope('train_step'):
    train_step = tf.train.AdagradOptimizer(FLAGS.lr).minimize(cross_entropy)
    #train_step= tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        mnist = input_data.read_data_sets(data_dir, one_hot=True)
        #prediction_result = []
        #accuracy_result=[]

        for i in range(FLAGS.epoch):
            if FLAGS.restore:
                saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
            else:
                train_image_batch, train_label_batch = mnist.train.next_batch(FLAGS.batch_size)
                fc2_, prediction,_=sess.run([fc2, y, train_step], feed_dict={x: train_image_batch, y_: train_label_batch, keep_prob: 0.5})
                """
                str_pre=''
                log = []
                for j in range(128):
                    str_pre +=str(fc2_[j])+'\n'
                    str_pre +=str(np.round(prediction[j],decimals=4)) + '\n'
                log.append(str_pre)
                tfg.saveEvalData('iter'+str(i)+'_mnist.txt',log)
                """
            if i % 10 == 0 or FLAGS.restore:
                test_image_batch, test_label_batch = mnist.test.next_batch(FLAGS.test_size)
                correct_prediction_result, accuracy_rate =sess.run([correct_prediction, accuracy], feed_dict={x: test_image_batch, y_: test_label_batch, keep_prob: 1.0})
                #str_pre=''
                #for item in correct_prediction_result:
                #    str_pre += str(item)+','
                #prediction_result.append(str_pre)
                #accuracy_result.append(accuracy_rate)
                print('iter:' + str(i), str(round(accuracy_rate*100,2))+'%')
                #print('iter:' + str_pre)

            if i > 0 and i % 100 == 0 and not FLAGS.restore:  #保存checkpoint
                saver.save(sess, ckpt_dir, global_step=i)
        """
        timelabel = time.strftime('%H%M%S', time.localtime())        
        tfg.saveEvalData('./prediction_result_'+timelabel +'.txt',prediction_result)
        tfg.saveEvalData('./accuracy_result_'+timelabel +'.txt',accuracy_result)
        """