#!/usr/bin/env python
# coding=utf-8
import contextlib
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.data import Dataset

tf.enable_eager_execution()

# 可配置区域开始
SPEAKER_NUMBER = 10
DEV_RATE = 0.1
TEST_RATE = 0.1
LEFT = 30
RIGHT = 10


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
# ###########################################################
with contextlib.closing(open('config')) as f:
    SRC, LOGDIR = f.read().split()

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
    full_dataset = full_dataset.interleave(build_one_dataset,
                                           cycle_length=SPEAKER_NUMBER,
                                           num_parallel_calls=8)
    full_dataset = full_dataset.batch(32)
    return full_dataset


def dense_model(features, labels, mode, params):
    net = tf.reshape(features, [-1, params['feature_dims']])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    predicted_classes = tf.argmax(logits, 1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


classifier = tf.estimator.Estimator(
    model_fn=dense_model,
    model_dir=LOGDIR,
    params={
        'feature_dims': 41*39,
        'hidden_units': [512, 512, 512, 512],
        'n_classes': 10,
    })

classifier.train(input_fn=lambda: build_data(train_set))
result = classifier.evaluate(input_fn=lambda: build_data(test_set))
