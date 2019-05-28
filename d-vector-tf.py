#!/usr/bin/env python
# coding=utf-8
import contextlib
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.data import Dataset
from models import *

tf.enable_eager_execution()

# 可配置区域开始
DEV_RATE = 0.1
TEST_RATE = 0.1
LEFT = 30
RIGHT = 10
EPCHO = 20
BATCH = 256


def EXTRACT(file):
    x = np.load(file)
    features = []
    for i in range(LEFT, len(x)-RIGHT):
        features.append(x[i-LEFT:i+RIGHT+1])
    features = np.reshape(features, (-1, (LEFT+RIGHT+1)*39))
    return features
# 可配置区域结束


# 读取配置
# 配置demo ##################################################
# /home/sub18/code/LibriSpeechFeature/simpleMFCC39.list
# /home/sub18/code/log/d-vector-pretrain
# 10
# ###########################################################
with contextlib.closing(open('config')) as f:
    SRC, LOGDIR, SPEAKER_NUMBER = f.read().split()
    SPEAKER_NUMBER = int(SPEAKER_NUMBER)

with contextlib.closing(open(SRC)) as f:
    files = f.read().split()
    ids = dict()
    for file in files:
        cid = int(os.path.basename(file).split('-')[0])
        if cid in ids:
            ids[cid].append(file)
        elif len(ids) < SPEAKER_NUMBER:
            ids[cid] = []

files_set = list(ids.values())
train_set, dev_set, test_set = [], [], []
for files in files_set:
    tmp_a = int(len(files)*(1-DEV_RATE-TEST_RATE))
    tmp_b = int(len(files)*(1-TEST_RATE))
    train_set.append(files[:tmp_a])
    dev_set.append(files[tmp_a:tmp_b])
    test_set.append(files[tmp_b:])


def build_data(data):

    def build_one_dataset(files, y):

        dataset = Dataset.from_tensor_slices(files)
        dataset = dataset.map(lambda x: tf.py_func(
            EXTRACT, inp=[x], Tout=[tf.float64]), num_parallel_calls=4)
        dataset = dataset.flat_map(lambda x: Dataset.from_tensor_slices(x))
        dataset = dataset.map(lambda x: (x, y))
        return dataset

    base_dataset = Dataset.range(SPEAKER_NUMBER)
    full_dataset = base_dataset.map(lambda x: tf.py_func(
        lambda x: [data[x]], inp=[x], Tout=[tf.string]))
    full_dataset = Dataset.zip((full_dataset, Dataset.range(SPEAKER_NUMBER)))
    full_dataset = full_dataset.interleave(build_one_dataset, cycle_length=SPEAKER_NUMBER,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
    full_dataset = full_dataset.shuffle(BATCH*4)
    full_dataset = full_dataset.batch(BATCH)
    return full_dataset


classifier = tf.estimator.Estimator(
    model_fn=dense_model,
    model_dir=LOGDIR,
    params={
        'feature_dims': 41*39,
        'hidden_units': [512, 512, 512, 512],
        'n_classes': SPEAKER_NUMBER,
    })

for i in range(EPCHO):
    classifier.train(input_fn=lambda: build_data(train_set))
    classifier.evaluate(input_fn=lambda: build_data(dev_set))
r=classifier.evaluate(input_fn=lambda:build_data(test_set))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**r))
