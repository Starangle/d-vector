import tensorflow as tf
from layers import Maxout

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
    optimizer = tf.train.AdagradOptimizer(0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def maxout_graph(inputs):
    net=tf.layers.Dense(256,activation=tf.nn.relu)(inputs)
    net=tf.layers.Dense(256,activation=tf.nn.relu)(net)
    net=tf.contrib.layers.maxout(net,256)
    net=tf.layers.Dropout(0.5)(net)
    net=tf.contrib.layers.maxout(net,256)
    net=tf.layers.Dropout(0.5)(net)
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






