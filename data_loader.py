#!/usr/bin/env python
# coding=utf-8
import contextlib
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

def converter(npy_file_path):
    mfcc=np.load(npy_file_path)
    l=mfcc.shape[0]
    context_mfcc=[]
    for i in range(l-40):
        context_mfcc.append(mfcc[i:i+41,:])
    tmp=np.reshape(context_mfcc,[-1,40*41])
    np.random.shuffle(tmp)
    return tmp

def load_data(cata,speaker_number,dev_rate=0.1,test_rate=0.1):
    with contextlib.closing(open(cata)) as f:
        files = f.read().split()
        ids = dict()
        for file in files:
            cid = int(os.path.basename(file).split('-')[0])
            if cid in ids:
                ids[cid].append(file)
            elif len(ids) < speaker_number:
                ids[cid] = []
            else:
                break

    files_set = list(ids.values())
    train_set, dev_set, test_set = [], [], []
    for files in files_set:
        tmp_a = int(len(files)*(1-dev_rate-test_rate))
        tmp_b = int(len(files)*(1-test_rate))
        train_set.append(files[:tmp_a])
        dev_set.append(files[tmp_a:tmp_b])
        test_set.append(files[tmp_b:])
    return train_set,dev_set,test_set

def build_data(data,batch,shuffle=0,repeat=False):

    def build_one_dataset(files, y):
        dataset = Dataset.from_tensor_slices(files)
        if shuffle!=0:
            dataset = dataset.shuffle(shuffle)
        dataset = dataset.map(lambda x: tf.py_func(
            converter, inp=[x], Tout=[tf.float64]), num_parallel_calls=4)
        dataset = dataset.flat_map(lambda x: Dataset.from_tensor_slices(x))
        dataset = dataset.map(lambda x: (x, y))
        return dataset
    
    speaker_number=len(data)
    base_dataset = Dataset.range(speaker_number)
    if repeat==True:
        base_dataset=base_dataset.repeat()
    full_dataset = base_dataset.map(lambda x: tf.py_func(
        lambda x: [data[x],[x]], inp=[x], Tout=[tf.string,tf.int64]))
    full_dataset = full_dataset.interleave(build_one_dataset, cycle_length=speaker_number,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
    full_dataset = full_dataset.shuffle(batch*128)
    full_dataset = full_dataset.batch(batch)
    return full_dataset

def verify_data(cata):
    SPLIT_RATE=0.2
    with contextlib.closing(open(cata)) as f:
        files = f.read().split()
        ids = dict()
        for file in files:
            cid = int(os.path.basename(file).split('-')[0])
            if cid in ids:
                ids[cid].append(file)
            else:
                ids[cid] = []
        files_set = list(ids.values())
        np.random.shuffle(files_set)
        target=files_set[0]
        attacker=np.hstack(files_set[1:])
        attacker=np.random.choice(attacker,len(target)).tolist()
        
        i=int(len(target)*SPLIT_RATE)
        train_set=[attacker[:i],target[:i]]
        test_set=[attacker[i:],target[i:]]
        return train_set,test_set

