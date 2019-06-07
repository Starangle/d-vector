#!/usr/bin/env python
# coding=utf-8
import contextlib
import os
import random
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from models import *
from data_loader import load_data,build_data

if __name__=='__main__':

    with contextlib.closing(open(sys.argv[1])) as f:
        config=json.load(f)
    
    counter=tf.Variable(0,name='global_step')

    drop_rate=tf.placeholder(tf.float32,shape=[1],name='drop_rate')
    inputs=tf.placeholder(dtype=tf.float64,shape=[None,1640],name='x')
    labels=tf.placeholder(dtype=tf.int64,shape=[None,1],name='y')

    if config['model']=='maxout':
        with tf.variable_scope('maxout'):
            net=maxout_graph(inputs,drop_rate)
            
    logits=tf.layers.Dense(int(config['speaker_number']))(net)
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    optimizer=tf.train.AdamOptimizer(0.01).minimize(loss,global_step=counter)
    init=tf.global_variables_initializer()

    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess,config['model_name'],global_step=counter)






