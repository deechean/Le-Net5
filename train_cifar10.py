#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 22:08:46 2019

@author: Deechean
"""
from cifar10 import cifar10
import os
import tensorflow as tf
from lenet5 import LetNet5
import tf_general as tfg
import numpy as np


FLAGS = tf.flags.FLAGS
try:
    tf.flags.DEFINE_integer('epoch', 8000, 'epoch')
    tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
    tf.flags.DEFINE_integer('test_size', 10000, 'test size')
    tf.flags.DEFINE_float('lr', 0.001, 'learning rate')
    tf.flags.DEFINE_boolean('restore', False, 'restore from checkpoint and run test')
except:
    print('parameters have been defined.')
 
data = cifar10();
train_image, train_label = data.get_train_batch(FLAGS.batch_size)
test_image, test_label = data.get_test_batch(FLAGS.test_size)

ckpt_dir = 'ckpt/'

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

with tf.name_scope('input'):
    x_image = tf.reshape(train_image, [-1, 32, 32,3])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    y_ = tf.cast(train_label, tf.float32)

with tf.name_scope('prediction'):
    le_net5 = LetNet5(x_image, keep_prob)
    fc2 = le_net5.fc2
    logits = le_net5.logits
    y = le_net5.prediction

with tf.name_scope('cross_entropy'):
    logy = tf.log(y)
    y_logy = y*logy
    reduce_sum = tf.reduce_sum(y_logy)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
    
    
with tf.name_scope('train_step'):
    train_step = tf.train.AdagradOptimizer(FLAGS.lr).minimize(cross_entropy)
    #train_step= tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(cross_entropy)

saver=tf.train.Saver(max_to_keep = 5)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        for i in range(FLAGS.epoch):
            #if FLAGS.restore:
            #    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
            #else:
            train_image_batch, train_label_batch = sess.run([train_image, train_label])
            
            #train_image_batch, train_label_batch = cifar10.get_batch(FLAGS.batch_size,train_data, train_label)
            #l, output, label_input, loss, _ =sess.run([logit, y, y_, cross_entropy, train_step], feed_dict={x: train_image_batch, y_: train_label_batch, keep_prob: 0.5})
            #tfg.printimages(train_image_batch)
            fc_2, logits_,prediction,labels,logy_, y_logy_,loss, _ , accuracy_rate = sess.run([fc2, logits, y, y_, logy, y_logy, cross_entropy,train_step,accuracy], 
                                                   feed_dict={keep_prob: 0.5}) 
            
            str_pre=''
            log = []
            for j in range(FLAGS.batch_size):
                str_pre += str(train_label_batch)
                """
                str_pre += str(fcnorm_[j])+'\n'
                #str_pre +=str(logits_[j])+'\n'
                str_pre +=str(np.round(prediction[j],decimals=2)) + '\n'
                str_pre +=str(labels[j]) + '\n'
                str_pre += str(logy_[j]) + '\n'
                str_pre += str(y_logy_[j]) + '\n'
                str_pre += str(loss)+ '\n'
                """
                str_pre += '--------------------------'
            log.append(str_pre)
            print('iter:' + str(i), 'fc2='+str(log))
            #tfg.saveEvalData('./cifar_log/iter'+str(i)+'_cifar.txt',log)
            
            
            if i % 10 == 0:  #保存预测模型
                saver.save(sess,'ckpt/cifar10_'+str(i)+'.ckpt',global_step=i)  
                print('iter:' + str(i), str(round(accuracy_rate*100,2))+'%')
        
        le_net5.input = tf.reshape(test_image, [-1, 32, 32,3])
        accuracy_rate = sess.run(accuracy)
        print('Test accuracy:', str(round(accuracy_rate*100,2))+'%')
           
        tf.reset_default_graph()