#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:23:43 2019

@author: Deechean
"""

import pickle
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import time 

def load(file_name):
    with open(file_name, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        return data  

class cifar10(object): 
    def __init__(self):
        self.train_images, self.train_labels = self._get_train()
        self.test_images, self.test_labels = self._get_test()
        
    def _get_train(self):
        train_labels = []        
        data1 = load('cifar-10-batches-py/data_batch_1')
        x1 = np.array(data1[b'data'])
        #x1 = x1.reshape(-1, 3, 32, 32).transpose(0,3,2,1).reshape(-1,32*32*3)
        y1 = data1[b'labels']
        train_data = np.array(x1)
        labels = np.array(y1)
        
        data2 = load('cifar-10-batches-py/data_batch_2')
        x2 = np.array(data2[b'data'])
        #x2 = x2.reshape(-1, 3, 32, 32).transpose(0,3,2,1).reshape(-1,32*32*3)
        y2 = data2[b'labels']
        train_data = np.append(train_data, x2)
        labels = np.append(labels, y2)

        data3 = load('cifar-10-batches-py/data_batch_3')
        x3 = np.array(data3[b'data'])
        #x3 = x3.reshape(-1, 3, 32, 32).transpose(0,3,2,1).reshape(-1,32*32*3)
        y3 = np.array(data3[b'labels']).reshape(10000)
        train_data = np.append(train_data, x3)
        labels = np.append(labels, y3)

        data4 = load('cifar-10-batches-py/data_batch_4')
        x4 = np.array(data4[b'data'])
        #x4 = x4.reshape(-1, 3, 32, 32).transpose(0,3,2,1).reshape(-1,32*32*3)
        y4 = np.array(data4[b'labels']).reshape(10000)
        train_data = np.append(train_data, x4)
        labels = np.append(labels, y4)
        
        #train_data.dtype = 'float32'
        train_data = train_data.reshape(-1, 3, 32, 32).transpose(0,3,2,1)
        
        #train_data, train_labels= self._append_distort_images(train_data, train_labels)
        
        for item in labels:
            train_labels.append(convert_label(item))
        #print('image shape:',np.shape(train_data))
        #print('label shape:',np.shape(train_labels))  
        if len(train_data) != len(train_labels):
            assert('train images ' + str(len(train_data))+' doesnt equal to train labels' + str(len(train_labels)))
            
        print('train set length: '+str(len(train_data)))
        return train_data, train_labels
 
    def _get_test(self):
        test_labels = list()
        data1 = load('cifar-10-batches-py/test_batch')
        x = np.array(data1[b'data'])
        x = x.reshape(-1, 3, 32, 32).transpose(0,3,2,1).reshape(-1,32*32*3)
        y = data1[b'labels']
        for item in y:
            test_labels.append(convert_label(item))
        #print('test image shape:',np.shape(x))
        #print('test label shape:',np.shape(test_labels))        
        print('test set length: '+str(len(x)))
        return x, test_labels
    
    def _distort_image(self, image):
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        # Generate a single distorted bounding box.
        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.1)
        # Employ the bounding box to distort the image.
        return tf.slice(image, begin, size)
    
    def _distort_color(self, input_image, color_ordering=0):
        image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        #print('_distort_color: enter')
        if color_ordering == 0:
            image = tf.image.random_brightness(image,32.0/255.0)
            image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
            image = tf.image.random_hue(image,0.2)
            image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
            image = tf.image.random_brightness(image,32.0/255.0)
            image = tf.image.random_hue(image,0.2)
            image = tf.image.random_contrast(image,lower=0.5,upper=1.5)  
        #print('_distort_color: exit')
        #image = tf.image.convert_image_dtype(input_image, dtype=tf.uint8)
        return tf.clip_by_value(image,0.0,1.0)
    
    def _append_distort_images(self, images, labels):
        distort_images = []
        i = 0
        with tf.Session() as sess:
            for image in images:  
                if i%100 == 0:
                    print('distort image:',str(i))
                i += 1
                image.reshape(32,32,3)
                distort_image = self._distort_color(image, np.random.randint(2))
                img = sess.run(distort_image)
                #print('process image')
                distort_images = np.append(distort_images,img)
                #print('append image')
            images = np.append(images, distort_images)
            labels = np.append(labels, train_label)
        return images, labels
    
    def get_train_batch(self, batch_size=128):
        input_queue = tf.train.slice_input_producer([self.train_images, self.train_labels], shuffle=True)
        image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, \
                  num_threads=1, capacity=64)
        return image_batch, label_batch

    def get_test_batch(self, batch_size=10000):
        input_queue = tf.train.slice_input_producer([self.test_images, self.test_labels], shuffle=True)
        image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, \
                  num_threads=1, capacity=64)
        return image_batch, label_batch
    
 
"""
def get_batch_(batch_size, image, label):
    batch_image = list()
    batch_label = list()
    indexs = list()
    for i in range(batch_size):
        index = random.randint(0, len(image)-1)
        while index in indexs:
            index = random.randint(0, len(image)-1)
        d = list(image[index])
        batch_image.append(d)
        z = label[index]
        batch_label.append(convert_label(z))
        indexs.append(index)
    return batch_image, batch_label
""" 
def convert_label(item):
         if item == 0:
            return [1,0,0,0,0,0,0,0,0,0]
         elif item == 1:
            return[0,0,0,0,0,0,0,0,0,1]
         elif item == 2:
            return[0,0,0,0,0,0,0,0,1,0]
         elif item == 3:
            return[0,0,0,0,0,0,0,1,0,0]
         elif item == 4:
            return[0,0,0,0,0,0,1,0,0,0]
         elif item == 5:
            return[0,0,0,0,0,1,0,0,0,0]      
         elif item == 6:
            return[0,0,0,0,1,0,0,0,0,0]
         elif item == 7:
            return[0,0,0,1,0,0,0,0,0,0]   
         elif item == 8:
            return[0,0,1,0,0,0,0,0,0,0]   
         else:
            return[0,1,0,0,0,0,0,0,0,0]         

if __name__ == '__main__':
    data = cifar10()
    batch_size=10
    
    with tf.Graph().as_default() as g: 
        x, y = data.get_train_batch(batch_size)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            i = 0
            while (not coord.should_stop()) and i < 5:
                batch_x, batch_y = sess.run([x, y])
                print ('batch_x shape',np.shape(batch_x))
                i += 1
                for img in batch_x:
                    print('image shape',np.shape(img))
                    plt.imshow(np.array(img).reshape(32,32,3))
                    plt.show()
            
        