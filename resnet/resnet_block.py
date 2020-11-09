import tensorflow.contrib.slim as slim
import tensorflow as tf

class block(object):
    def __init__(self):
        return

    @staticmethod
    def _bottleneck_block(x, oc, name, trainable=True, weight_decay=0.0001, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            w, h, ic = x.get_shape()
            weight = tf.get_variable(name + '-filter-64d-1',
                                     shape=[oc['ksize'], oc['ksize'], ic, oc['channel']],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     trainable=trainable,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            o = tf.nn.conv2d(x, filter=weight, strides=[1, oc['stride'], oc['stride'],1], padding='SAME', name='-conv2d-64d-1')
            o = tf.nn.relu(o, name='-relu-64d-1')

            weight = tf.get_variable(name + '-filter-64d-2',
                                     shape=[oc['ksize'], oc['ksize'],  o.get_shape()[-1], oc['channel']],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     trainable=trainable,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            o = tf.nn.conv2d(o, filter=weight, strides=[1, oc['stride'], oc['stride'], 1], padding='SAME', name='-conv2d-64d-2')

            o = tf.add(x, o)
            o = tf.nn.relu(o, name='-relu-64d-2')
        return o

    @staticmethod
    def _bottleneck_block_11(x, oc, name, trainable=True, weight_decay=0.0001, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            w, h, ic = x.get_shape()
            weight = tf.get_variable(name + '-filter-256d-1',
                                     shape=[1, 1, ic, oc['channel1']],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     trainable=trainable,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            o = tf.nn.conv2d(x, filter=weight, strides=[1, 1, 1,1], padding='SAME', name='-conv2d-256d-1')
            o = tf.nn.relu(o, name='-relu-256d-1')

            weight = tf.get_variable(name + '-filter-256d-2',
                                     shape=[oc['ksize'], oc['ksize'], o.get_shape()[-1], oc['channel1']],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     trainable=trainable,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            o = tf.nn.conv2d(o, filter=weight, strides=[1, oc['stride'], oc['stride'],1], padding='SAME', name='-conv2d-256d-2')
            o = tf.nn.relu(o, name='-relu-256d-2')

            weight = tf.get_variable(name + '-filter-256d-3',
                                     shape=[1, 1,  o.get_shape()[-1], oc['channel2']],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     trainable=trainable,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            o = tf.nn.conv2d(o, filter=weight, strides=[1, 1, 1,1], padding='SAME', name='-conv2d-256d-3')

            o = tf.add(x, o)
            o = tf.nn.relu(o, name='-relu-256d-3')
        return o

