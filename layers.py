#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

class Maxout(tf.keras.layers.Layer):

    def __init__(self,output_dim,k,**kwargs):
        self.output_dim=output_dim
        self.k=k
        super(Maxout,self).__init__(**kwargs)

    def build(self, input_shape):
        wshape = tf.TensorShape((self.k, input_shape[1], self.output_dim))
        bshape=tf.TensorShape((self.k,self.output_dim))
        self.weights = self.add_weight(name='weights',shape=wshape,
                                      initializer='uniform',trainable=True)
        self.bias=self.add_weight(name='bias',shape=bshape
                                 initializer='uniform',trainable=True)
        super(Maxout, self).build(input_shape)

    def call(self, inputs):
        return tf.maximum(tf.tensordot(inputs,self.weights)+self.b,axis=1)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(Maxout, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
