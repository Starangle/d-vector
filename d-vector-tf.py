#!/usr/bin/env python
# coding=utf-8
import contextlib,os,random
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.data import Dataset

tf.enable_eager_execution()

# 可配置区域开始
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
SRC='/home/sub18/code/LibriSpeechFeature/simpleMFCC39.list'
LOGDIR='/home/sub18/code/log/d-vector-pretrain'
SPEAKER_NUMBER=10
DEV_RATE=0.1
TEST_RATE=0.1
LEFT=30
RIGHT=10

def EXTRACT(file):
    x=np.load(file)
    features=[]
    print(type(features))
    for i in range(LEFT,len(x)-RIGHT):
        features.append(x[i-LEFT:i+RIGHT+1])
    features=np.reshape(features,(-1,(LEFT+RIGHT+1)*39))
    return features

# 可配置区域结束

with contextlib.closing(open(SRC)) as f:
    files=f.read().split()

    ids=dict()
    for file in files:
        cid=int(os.path.basename(file).split('-')[0])
        if cid in ids:
            ids[cid].append(file)
        elif len(ids)<SPEAKER_NUMBER:
            ids[cid]=[]

files_set=list(ids.values())
train_set,dev_set,test_set=[],[],[]
for files in files_set:
    tmp_a=int(len(files)*(1-DEV_RATE-TEST_RATE))
    tmp_b=int(len(files)*(1-TEST_RATE))
    train_set.append(files[:tmp_a])
    dev_set.append(files[tmp_a:tmp_b])
    test_set.append(files[tmp_b:])
    
def build_data(data):
    
    def get_id(x):
        return [data[x]]
        
    def build_one_dataset(files,y):
        
        def get_one_hot(y):
            return to_categorical(y,num_classes=SPEAKER_NUMBER)
        
        label=tf.py_func(get_one_hot,inp=[y],Tout=[tf.float32])
        dataset=Dataset.from_tensor_slices(files)
        dataset=dataset.map(lambda x:tf.py_func(EXTRACT,inp=[x],Tout=[tf.float64]),num_parallel_calls=4)
        dataset=dataset.flat_map(lambda x:Dataset.from_tensor_slices(x))
        dataset=dataset.map(lambda x:(x,label))
        return dataset
    
    base_dataset=Dataset.range(SPEAKER_NUMBER)
    full_dataset=base_dataset.map(lambda x:tf.py_func(get_id,inp=[x],Tout=[tf.string]))
    full_dataset=Dataset.zip((full_dataset,Dataset.range(SPEAKER_NUMBER)))
    full_dataset=full_dataset.interleave(build_one_dataset,
                                         cycle_length=SPEAKER_NUMBER,
                                         num_parallel_calls=8)
    return full_dataset 

train_dataset=build_data(train_set)
for e in tf.contrib.eager.Iterator(train_dataset):
    print(e)

def dense_model(features,labels,mode,params):
    net=features
    for unit in params['hidden_units']:
        net=tf.layers.dense(net,units=units,activation=tf.nn.relu)

    logits=tf.layers.dense(net,params['n_class'],activation=tf.nn.softmax)

    loss=tf.losses.softmax_cross_entropy()
