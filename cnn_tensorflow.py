# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 21:22:28 2016

@author: Aaron
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define CNN structure
def CNN_frame(xs):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def conv2d(x, shape, activation=None):
        W = weight_variable(shape)
        b = weight_variable([shape[-1]])
        W_plus_b =  tf.add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'), b)
        if activation == None:
            return W_plus_b
        else:
            return activation(W_plus_b)

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def full_connected(x, shape, activation=None):
        x = tf.reshape(x, [-1, shape[0]])
        W = weight_variable(shape)
        b = weight_variable([shape[-1]])
        W_plus_b = tf.add(tf.matmul(x, W), b)
        if activation == None:
            return W_plus_b
        else:
            return activation(W_plus_b)

    def dropout_feature(x, keep_prob):
        return tf.nn.dropout(x, keep_prob)
    
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    
    conv1 = conv2d(x_image, [5,5,1,32], activation=tf.nn.relu) # output size 28x28x32
    pool1 = max_pool_2x2(conv1)
    
    conv2 = conv2d(pool1, [5,5, 32, 64], activation=tf.nn.relu) # output size 14x14x64
    pool2 = max_pool_2x2(conv2)
    
    fc1 = full_connected(pool2, [7*7*64, 1024], activation=tf.nn.relu)
    fc1_drop = dropout_feature(fc1, keep_prob)
    
    prediction = full_connected(fc1_drop, [1024, 10], activation=tf.nn.softmax)
    
    return prediction

# train the model
def model_fit(xs, ys): 
    
    def compute_accuracy(v_xs, v_ys):
        y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 0.8})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 0.8})
        return result
    
    prediction = CNN_frame(xs)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))   
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        epochs = 1
        for epoch in range(epochs):
            print('Epoch: ', epoch+1)
            for i in range(1000):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.8})
                if i % 50 == 0:
                    print('Accuracy: ', compute_accuracy(mnist.test.images, mnist.test.labels))
                    
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32) 

# train the model
model_fit(xs, ys)