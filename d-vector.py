#!/usr/bin/env python
# coding=utf-8
from sklearn.metrics import roc_curve
import contextlib
import os
import random
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from data_loader import *
from layers import Maxout


def maxout_model(inputs, drop_rate):
    net = tf.layers.Dense(256, activation=tf.nn.relu)(inputs)
    net = tf.layers.Dense(256, activation=tf.nn.relu)(net)
    net = tf.contrib.layers.maxout(net, 256)
    net = tf.layers.Dropout(drop_rate)(net)
    net = tf.contrib.layers.maxout(net, 256)
    net = tf.layers.Dropout(drop_rate)(net)
    return net


def dnn_model(inputs):
    net = tf.layers.Dense(256, activation=tf.nn.relu)(inputs)
    net = tf.layers.Dense(256, activation=tf.nn.relu)(net)
    net = tf.layers.Dense(256, activation=tf.nn.relu)(net)
    net = tf.layers.Dense(256, activation=tf.nn.relu)(net)
    return net


def cnn_model(inputs):
    reshaped_inputs = tf.reshape(inputs, [None, 41, 40, 1])
    net = tf.layers.Conv2D(32, [5, 5], padding='same')(reshaped_inputs)
    net = tf.layers.MaxPooling2D(
        pool_size=[2, 2], strides=2, padding='same')(net)
    net = tf.layers.Conv2D(64, [5, 5], padding='same')(net)
    net = tf.layers.MaxPooling2D(
        pool_size=[2, 2], strides=2, padding='same')(net)
    net = tf.reshape(net, [None, 11*10*64])
    net = tf.layers.Dense(1024, activation=tf.nn.relu)(net)
    net = tf.layers.Dense(600, activation=tf.nn.relu)(net)
    return net


def create(config):
    global_step = tf.Variable(0, name='global_step')
    drop_rate = tf.placeholder(tf.float32, shape=None, name='drop_rate')

    inputs = tf.placeholder(dtype=tf.float64, shape=[None, 1640], name='x')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='y')
    if config['model'] == 'maxout':
        with tf.variable_scope('maxout'):
            net = maxout_model(inputs, drop_rate)
    elif config['model'] == 'dnn':
        with tf.variable_scope('dnn'):
            net = dnn_model(inputs)
    logits = tf.layers.Dense(int(config['speaker_number']),name='logits')(net)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    opti = tf.train.AdagradOptimizer(0.01).minimize(loss,global_step=global_step,name='optimizer')
    loss_summary=tf.summary.scalar('loss',loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, config['model_name'])


def train(config, train_set, dev_set):
    train_iter = build_data(
        train_set, int(config['batch']), repeat=True).make_one_shot_iterator().get_next()
    dev_iter = build_data(
        dev_set, int(config['batch']), repeat=True).make_one_shot_iterator().get_next()

    model = tf.train.get_checkpoint_state(config['model_dir'])
    saver = tf.train.import_meta_graph(config['model_name']+'.meta')
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name('x:0')
    labels = graph.get_tensor_by_name('y:0')
    global_step = graph.get_tensor_by_name('global_step:0')
    drop_rate = graph.get_tensor_by_name('drop_rate:0')
    logits = graph.get_tensor_by_name('logits/BiasAdd:0')
    loss_summary=graph.get_tensor_by_name('loss:0')
    opti=graph.get_tensor_by_name('optimizer:0')

    predictions=tf.argmax(logits,1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(labels,predictions),tf.float32))
    stream_accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions, name='stream_accuracy')
    stream_accuracy_init = tf.variables_initializer(var_list=tf.get_collection(
        tf.GraphKeys.LOCAL_VARIABLES, scope='stream_accuracy'))
    init=tf.global_variables_initializer()

    batch_acc_summary=tf.summary.scalar('accuracy',accuracy)
    stream_acc_summary=tf.summary.scalar('global_accuracy',stream_accuracy[1])
    train_summary=tf.summary.merge(
        inputs=[
            loss_summary,batch_acc_summary
        ]
    )
    dev_summary=tf.summary.merge(
        inputs=[
            stream_acc_summary,
        ]
    )

    train_writer = tf.summary.FileWriter(config['train_dir'])
    dev_writer = tf.summary.FileWriter(config['dev_dir'])

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model.model_checkpoint_path)

        for epoch in range(int(config['epoch'])):
            for i in range(int(config['speaker_number'])*100):
                x, y = sess.run(train_iter)
                y=np.reshape(y,[-1])
                result = sess.run([opti,train_summary], feed_dict={
                    inputs: x, labels: y, drop_rate: 0.5
                })
                train_writer.add_summary(result[-1], global_step=global_step.eval())

            sess.run(stream_accuracy_init)
            for i in range(int(config['speaker_number'])*10):
                x, y = sess.run(dev_iter)
                y=np.reshape(y,[-1])
                result = sess.run([dev_summary], feed_dict={
                    inputs: x, labels: y, drop_rate: 0
                })
            dev_writer.add_summary(result[-1],global_step=global_step.eval())
        saver.save(sess,config['model_name'],global_step=global_step,write_meta_graph=False)

def verify(config):
    model=tf.train.get_checkpoint_state(config['model_dir'])
    saver=tf.train.import_meta_graph(config['model_name']+'.meta')
    graph=tf.get_default_graph()
    inputs=graph.get_tensor_by_name('x:0')
    logits=graph.get_tensor_by_name('logits/BiasAdd:0')
    drop_rate=graph.get_tensor_by_name('drop_rate:0')
    new_labels=tf.placeholder(tf.int64,shape=[None])

    with tf.variable_scope('verify'):
        net=tf.layers.Dense(128,activation=tf.nn.relu)(logits)
        net=tf.layers.Dense(32,activation=tf.nn.relu)(net)
        new_logits=tf.layers.Dense(1,activation=tf.nn.sigmoid)(net)
    new_loss=tf.reduce_mean(tf.square(new_logits-tf.cast(new_labels,tf.float64)))
    new_optimizer=tf.train.AdagradOptimizer(0.01).minimize(
        new_loss,var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,scope='verify')
    )

    train_set,test_set=verify_data(config['verify_src'])
    train_iter = build_data(
        train_set, int(config['batch']), repeat=True).make_one_shot_iterator().get_next()
    test_iter = build_data(
        test_set, int(config['batch'])).make_one_shot_iterator().get_next()

    new_init=tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(new_init)
        saver.restore(sess,model.model_checkpoint_path)

        for i in range(int(config['verify_train_epoch'])):
            x,y=sess.run(train_iter)
            y=np.reshape(y,[-1])
            sess.run([new_optimizer,new_loss],feed_dict={
                inputs:x,new_labels:y,drop_rate:0
            })
        

        scores,gt=[],[]
        try:
            while 1:
                x,y=sess.run(test_iter)
                y=np.reshape(y,[-1])
                result=sess.run(new_logits,feed_dict={
                    inputs:x,new_labels:y,drop_rate:0
                })
                scores.extend(result)
                gt.extend(y)
        except:
            fpr, tpr, threshold = roc_curve(gt,scores, pos_label=1)
            fnr = 1 - tpr
            eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
            print(eer,eer_threshold)


if __name__ == '__main__':

    cmd = sys.argv[1]
    with contextlib.closing(open(sys.argv[2])) as f:
        config = json.load(f)

    train_set, dev_set, test_set = load_data(config['src'],
                                             int(config['speaker_number']),
                                             float(config['dev_rate']),
                                             float(config['test_rate']))
    if cmd == 'create':
        create(config)
    elif cmd == 'train':
        train(config, train_set, dev_set)
    elif cmd == 'test':
        test(config)
    elif cmd =='verify':
        verify(config)
    else:
        print("Unknown command!")
