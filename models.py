import tensorflow as tf
from layers import Maxout

def maxout_graph(inputs,drop_rate):
    net=tf.layers.Dense(256,activation=tf.nn.relu)(inputs)
    net=tf.layers.Dense(256,activation=tf.nn.relu)(net)
    net=tf.contrib.layers.maxout(net,256)
    net=tf.layers.Dropout(drop_rate)(net)
    net=tf.contrib.layers.maxout(net,256)
    net=tf.layers.Dropout(drop_rate)(net)
    return net

def dense_graph(inputs):
    net=tf.layers.Dense(512,activation=tf.nn.relu)(inputs)
    net=tf.layers.Dense(512,activation=tf.nn.relu)(net)
    net=tf.layers.Dense(512,activation=tf.nn.relu)(net)
    net=tf.layers.Dense(512,activation=tf.nn.relu)(net)
    return net

def cnn_graph(inputs):
    reshaped_inputs=tf.reshape(inputs,[None,41,40,1])
    net=tf.layers.Conv2D(32,[5,5],padding='same')(reshaped_inputs)
    net=tf.layers.MaxPooling2D(pool_size=[2,2],strides=2,padding='same')(net)
    net=tf.layers.Conv2D(64,[5,5],padding='same')(net)
    net=tf.layers.MaxPooling2D(pool_size=[2,2],strides=2,padding='same')(net)
    net=tf.reshape(net,[None,11*10*64])
    net=tf.layers.Dense(1024,activation=tf.nn.relu)(net)
    net=tf.layers.Dense(600,activation=tf.nn.relu)(net)
    return net






