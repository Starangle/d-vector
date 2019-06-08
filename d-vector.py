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
from data_loader import load_data, build_data
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
    drop_rate = tf.placeholder(tf.float32, shape=[1], name='drop_rate')

    inputs = tf.placeholder(dtype=tf.float64, shape=[None, 1640], name='x')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='y')
    inner_labels = tf.reshape(labels, [-1])
    if config['model'] == 'maxout':
        with tf.variable_scope('maxout'):
            net = maxout_model(inputs, drop_rate)
    elif config['model'] == 'dnn':
        with tf.variable_scope('dnn'):
            net = dnn_model(inputs)
    logits = tf.layers.Dense(int(config['speaker_number']))(net, name='logits')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, config['model_name'], global_step=global_step)


def train(config, train_set, test_set):
    train_iter = build_data(
        train_set, BATCH, repeat=True).make_one_shot_iterator().get_next()
    dev_iter = build_data(
        dev_set, BATCH, repeat=True).make_one_shot_iterator().get_next()

    model = tf.train.get_checkpoint_state(config['model_dir'])
    saver = tf.train.import_meta_graph(model.model_checkpoint_path+'.meta')
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name('inputs')
    labels = graph.get_tensor_by_name('labels')
    global_step = graph.get_tensor_by_name('global_step')
    drop_rate = graph.get_tensor_by_name('drop_rate')
    logits = graph.get_tensor_by_name('logits')

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    opti = tf.train.AdamOptimizer(0.01).minimize(loss)
    accuracy = tf.reduce_mean(
        tf.equal(labels-tf.argmax(logits, 1)))
    stream_accuracy = tf.metrics.accuracy(
        labels=labels, logits=logits, name='stream_accuracy')
    stream_accuracy_init = tf.variables_initializer(var_list=tf.get_collection(
        tf.GraphKeys.LOCAL_VARIABLES, scope='stream_accuracy'))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('accuracy', stream_accuracy)
    summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(config['train_dir'])
    dev_writer = tf.summary.FileWriter(config['dev_dir'])

    with tf.Session() as sess:
        saver.restore(sess, model.model_checkpoint_path)

        for epoch in range(int(config['epoch'])):
            for i in range(int(config['speaker_number'])*100):
                x, y = sess.run(train_iter)
                result = sess.run([opti, accuarcy, summary], feed_dict={
                    inputs: x, labels: y, drop_rate: 0.5
                })
                train_writer.add_summary(res[-1], global_step=global_step.eval())

            sess.run(stream_accuracy_init)
            for i in range(int(config['speaker_number'])*10):
                x, y = sess.run(dev_iter)
                result = sess.run([stream_accuracy, summary], feed_dict={
                    inputs: x, labels: y, drop_rate: 0
                })
            dev_writer.add_summary(res[-1],global_step=global_step.eval())
        saver.save(sess,config['model_name'],global_step=global_step)


if __name__ == '__main__':

    cmd = sys.argv[1]
    with contextlib.closing(open(sys.argv[2])) as f:
        config = json.load(f)

    train_set, dev_set, test_set = load_data(config['src'],
                                             config['speaker_number'],
                                             config['dev_rate'],
                                             config['test_rate'])
    if cmd == 'create':
        create(config)
    elif cmd == 'train':
        train(config, train_set, dev_test)
    elif cmd == 'test':
        test(config)
    else:
        print("Unknown command!")
