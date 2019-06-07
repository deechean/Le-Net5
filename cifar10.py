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
        self.train_indexs = list()
        self.test_indexs = list()
        self.train_images, self.train_labels = self._get_train()
        self.test_images, self.test_labels = self._get_test()
        self.label_dic = {0:'aircraft', 1:'car',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

        
    def _get_train(self):
        train_labels = []        
        data1 = load('cifar-10-batches-py/data_batch_1')
        x1 = np.array(data1[b'data'])
        #x1 = x1.reshape(-1, 3, 32, 32).transpose(0,3,2,1).reshape(-1,32*32*3)
        y1 = data1[b'labels']
        train_data = np.array(x1)
        train_labels = np.array(y1)
        
        data2 = load('cifar-10-batches-py/data_batch_2')
        x2 = np.array(data2[b'data'])
        #x2 = x2.reshape(-1, 3, 32, 32).transpose(0,3,2,1).reshape(-1,32*32*3)
        y2 = data2[b'labels']
        train_data = np.append(train_data, x2)
        train_labels = np.append(train_labels, y2)

        data3 = load('cifar-10-batches-py/data_batch_3')
        x3 = np.array(data3[b'data'])
        #x3 = x3.reshape(-1, 3, 32, 32).transpose(0,3,2,1).reshape(-1,32*32*3)
        y3 = np.array(data3[b'labels']).reshape(10000)
        train_data = np.append(train_data, x3)
        train_labels = np.append(train_labels, y3)

        data4 = load('cifar-10-batches-py/data_batch_4')
        x4 = np.array(data4[b'data'])
        #x4 = x4.reshape(-1, 3, 32, 32).transpose(0,3,2,1).reshape(-1,32*32*3)
        y4 = np.array(data4[b'labels']).reshape(10000)
        train_data = np.append(train_data, x4)
        train_labels = np.append(train_labels, y4)
        
        data5 = load('cifar-10-batches-py/data_batch_5')
        x5 = np.array(data4[b'data'])
        #x5 = x5.reshape(-1, 3, 32, 32).transpose(0,3,2,1).reshape(-1,32*32*3)
        y5 = np.array(data4[b'labels']).reshape(10000)
        train_data = np.append(train_data, x5)
        train_labels = np.append(train_labels, y5)
        
        train_data = train_data.reshape(-1, 3, 32, 32).transpose(0,3,2,1).reshape(-1,32,32,3)
        train_labels.astype(np.int64)
        #train_data, train_labels= self._append_distort_images(train_data, train_labels)
        
        #for item in labels:
        #    train_labels.append(item)
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
        x = x.reshape(-1, 3, 32, 32).transpose(0,3,2,1).reshape(-1,32,32,3)
        y = data1[b'labels']
        for item in y:
            test_labels.append(item)
        print('test image shape:',np.shape(x))
        print('test label shape:',np.shape(test_labels))        
        print('test set length: '+str(len(x)))
        return x, test_labels
    
    def _resize(self,image):
        resized_image = np.ndarray.reshape(image,(32,32,3))[2:30,2:30,0:3] 
        #print(resized_image.shape)
        return resized_image
    
    def random_flipper(self,image):
        if random.random() < 0.5:
            swap_time = int(len(image)/2)
            for i in range(swap_time):
                image[[i,len(image)-i-1],:] = image[[len(image)-i-1,i],:]
        return image
        
    def image_distort(self,image):
        
        return image
    
    def random_bright(self, image, delta=32):
        if random.random() < 0.5:
            delta_r = int(random.uniform(-delta, delta))
            delta_g = int(random.uniform(-delta, delta))
            delta_b = int(random.uniform(-delta, delta))
            image = image.transpose(2,1,0)
            #print(1)
            #print(np.shape(image))
            
            R = image[0] + delta_r
            G = image[0] + delta_g
            B = image[0] + delta_b
            
            image = np.asarray([R,G,B]).transpose(2,1,0) 
            #print(2)
            #print(np.shape(image))
            image = image.clip(min=0, max=255)
        return image
   
    def get_train_batch(self,batch_size=128):
        batch_image = list()
        batch_label = list()
        data_index = list()
        i = 0
        while i < batch_size:
            index = random.randint(0, len(self.train_images)-1)
            if not index in self.train_indexs:
                i += 1
                d = self.random_bright(self.random_flipper(self.train_images[index]))
                batch_image.append(self._resize(d))
                batch_label.append(self.train_labels[index])
                self.train_indexs.append(index)
                data_index.append(index)
                if len(self.train_indexs) >=  len(self.train_images):
                    self.train_indexs.clear()
        return batch_image, batch_label, data_index
        
    def get_test_batch(self,batch_size=10000):
        batch_image = list()
        batch_label = list()
        data_index = list()
        i = 0
        while i < batch_size:
            index = random.randint(0, len(self.test_images)-1)
            if not index in self.test_indexs:
                i += 1
                d = self.test_images[index]
                batch_image.append(self._resize(d)) 
                batch_label.append(self.test_labels[index])
                self.test_indexs.append(index)
                data_index.append(index)
                if len(self.test_indexs) >=  len(self.test_images):
                    self.test_indexs.clear()
        return batch_image, batch_label,data_index

    
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
                    img = data.image_flipper(img)
                    plt.imshow(np.array(img).reshape(32,32,3))
                    plt.show()
            
        