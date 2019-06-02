#!/usr/bin/env python
# coding=utf-8
import contextlib
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.data import Dataset
from models import maxout_graph
from data_loader import load_data,build_data

# 可配置区域开始
SRC='/home/sub18/code/data_catalogue/mfcc_clean_360.list'
TRAIN_LOG_DIR='/home/sub18/code/log/speaker_recognition/train'
DEV_LOG_DIR='/home/sub18/code/log/speaker_recognition/dev'
MODEL_DIR='/home/sub18/code/models/speaker_recognition'
MODEL_NAME='/home/sub18/code/models/speaker_recognition/speaker_recognition'
SPEAKER_NUMBER=500
DEV_RATE = 0.1
TEST_RATE = 0.1
LEFT = 30
RIGHT = 10
EPCHO = 10
BATCH = 512
FEATURE_DIM=41*40
# 可配置区域结束

# 数据准备
train_set, dev_set, test_set = load_data(SRC,SPEAKER_NUMBER,DEV_RATE,TEST_RATE)
train_iter=build_data(train_set,BATCH,repeat=True).make_one_shot_iterator().get_next()
dev_iter=build_data(dev_set,BATCH,repeat=True).make_one_shot_iterator().get_next()
test_iter=build_data(test_set,BATCH).make_one_shot_iterator().get_next()

# 计数器
counter=tf.Variable(0)

# 网络拓扑
inputs=tf.placeholder(dtype=tf.float64,shape=[None,FEATURE_DIM])
labels=tf.placeholder(dtype=tf.int64,shape=[None,1])
net=maxout_graph(inputs)
logits=tf.layers.Dense(SPEAKER_NUMBER)(net)

# 指标计算
loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
predict=tf.argmax(logits,1)
reshaped_label=tf.reshape(labels,[-1])
acc=tf.reduce_mean(tf.cast(tf.equal(predict,reshaped_label),dtype=tf.float64))
stream_acc=tf.metrics.accuracy(predictions=predict,labels=reshaped_label)

# 优化方案
optimizer=tf.train.AdagradOptimizer(0.01).minimize(loss,global_step=counter)

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







