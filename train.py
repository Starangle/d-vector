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

if __name__'__main__':

    with contextlib.closing(open(sys.argv[1])) as f:
        config=json.load(f)
    
    checkpoint=tf.train.get_checkpoint_state(config['model_dir'])
    saver=tf.train.import_meta_graph(checkpoint.model_checkpoint_path+'.meta')

    train_set, dev_set, test_set = load_data(config['src'],
                                             int(config['speaker_number']),
                                             float(config['dev_rate']),
                                             float(config['test_rate']))
    train_iter=build_data(train_set,BATCH,repeat=True).make_one_shot_iterator().get_next()
    dev_iter=build_data(dev_set,BATCH,repeat=True).make_one_shot_iterator().get_next()
    
    graph=tf.get_default_graph()
    inputs=graph.get_tensor_by_name('x')
    labels=tf.reshape(graph.get_tensor_by_name('y'),[-1])
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    predict=tf.argmax(logits,1)
    acc=tf.reduce_mean(tf.cast(tf.equal(predict,labels),dtype=tf.float64))

    # 优化方案
    optimizer=tf.train.AdamOptimizer(0.01).minimize(loss,global_step=counter)

    # 可视化
    tf.summary.scalar('loss',loss)
    tf.summary.scalar('acc',acc)
    summary=tf.summary.merge_all()
    train_writer=tf.summary.FileWriter(TRAIN_LOG_DIR,tf.get_default_graph())
    dev_writer=tf.summary.FileWriter(DEV_LOG_DIR,tf.get_default_graph())

    # 初始化图中已经存在的变量
    init=tf.global_variables_initializer()
    local_init=tf.local_variables_initializer()

    # 持久化
    checkpoint=tf.train.get_checkpoint_state(MODEL_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        model_path=checkpoint.model_checkpoint_path
    else:
        model_path=None
    saver=tf.train.Saver()

    with tf.Session() as sess:
        if model_path is not None:
            saver.restore(sess,checkpoint.model_checkpoint_path)
        else:
            sess.run(init)

        for epcho in range(EPCHO):
            for i in range(SPEAKER_NUMBER*100):
                x,y=sess.run(train_iter)
                res=sess.run([acc,optimizer,summary],feed_dict={inputs:x,labels:y})
                train_writer.add_summary(res[-1],global_step=counter.eval())
                
                # verify every SPEAKER_NUMBER steps
                if i%SPEAKER_NUMBER==0:
                    dx,dy=sess.run(dev_iter)
                    res=sess.run([acc,loss,summary],feed_dict={inputs:dx,labels:dy})
                    dev_writer.add_summary(res[-1],global_step=counter.eval())

                # save 10 times in one epcho
                if i%(SPEAKER_NUMBER*10)==0:
                    saver.save(sess,MODEL_NAME,global_step=counter)

        # test model
        test_acc,test_count=0,0
        try:
            while 1:
                tx , ty =sess.run(test_iter)
                res=sess.run([acc],feed_dict={inputs:tx,labels:ty})
                test_acc+=res[0]
                test_count+=1
        except:
            print(test_acc/test_count)







