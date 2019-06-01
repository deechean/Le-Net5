#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 00:02:11 2019

@author: Deechean
"""

from cifar10 import cifar10
import os
import tensorflow as tf
from lenet5 import LetNet5
import tf_general as tfg
import numpy as np
import matplotlib.pyplot as plt



def evaluate_lenet5(x):  
    le_net5 = LetNet5(x, 1)
    y = le_net5.prediction
    #return y
    return tf.argmax(y,1)

def load_image(image_name):
    return


def check_image(data):
    for i in range(len(data.test_images)):
        plt.imshow(data.test_images[i])
        plt.show()            
        print('Label: ' + data.label_dic[data.test_labels[i]])
        key = input('Press any keys to continue. Press q to quit. ')
        if key == 'q':
            break


if __name__ == '__main__':
    data = cifar10();
  
    ckpt_dir = 'ckpt/'
    model_file = tf.train.latest_checkpoint(ckpt_dir)
    
    x = tf.placeholder(tf.float32,[None, 28, 28,3],name='input')
    prediction = evaluate_lenet5(x)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess,model_file)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        for i in range(10000):
            test_image, test_label = data.get_test_batch(1)
            output= sess.run(prediction,feed_dict={x:test_image})
            plt.imshow(test_image[0])
            plt.show()            
            print('Label: ' + data.label_dic[test_label[0]])
            print('Prediction: '+ data.label_dic[output[0]])
            key = input('Press any keys to continue. Press q to quit. ')
            if key == 'q':
                break     


