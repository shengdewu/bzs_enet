import tensorflow.contrib.slim as slim
import tensorflow as tf

class resnet_block(object):
    def __init__(self):
        return

    @staticmethod
    def bottleneck_block(x, oc, name, trainable=True, weight_decay=0.0001, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            _, iw, ih, ic = list(x.get_shape())
            weight = tf.get_variable('filter-1',
                                     shape=[oc['ksize'], oc['ksize'], ic, oc['channel']],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     trainable=trainable,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            o = tf.nn.conv2d(x, filter=weight, strides=[1, oc['stride1'], oc['stride1'], 1], padding='SAME', name='-conv2d-1')
            o = slim.batch_norm(o, is_training=trainable, fused=True, scope='batch-norm-1')
            o = tf.nn.relu(o, name='relu-1')

            weight = tf.get_variable('filter-2',
                                     shape=[oc['ksize'], oc['ksize'],  o.get_shape()[-1], oc['channel']],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     trainable=trainable,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            o = tf.nn.conv2d(o, filter=weight, strides=[1, oc['stride2'], oc['stride2'], 1], padding='SAME', name='-conv2d-2')
            o = slim.batch_norm(o, is_training=trainable, fused=True, scope='batch-norm-2')

            _, ow, oh, oc = list(o.get_shape())
            if oc != ic:
                ox = x
                if ow != iw:
                    ox = slim.conv2d(x, ic, [3, 3], 2, padding='SAME', reuse=reuse, scope='add-dimensions-adjust')
                ox = slim.conv2d(ox, oc, [1, 1], stride=1, padding='SAME', reuse=reuse, scope='add-dimensions')
                o = tf.add(ox, o)
            else:
                o = tf.add(x, o)
            o = tf.nn.relu(o, name='relu-2')
        return o

    @staticmethod
    def bottleneck_block_11(x, oc, name, trainable=True, weight_decay=0.0001, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            _, iw, ih, ic = list(x.get_shape())
            weight = tf.get_variable('filter-11-1',
                                     shape=[1, 1, ic, oc['channel1']],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     trainable=trainable,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            o = tf.nn.conv2d(x, filter=weight, strides=[1, 1, 1, 1], padding='SAME', name='conv2d-11-1')
            o = slim.batch_norm(o, is_training=trainable, fused=True, scope='batch-norm-11-1')
            o = tf.nn.relu(o, name='relu-11-1')

            weight = tf.get_variable('filter-11-2',
                                     shape=[oc['ksize'], oc['ksize'], o.get_shape()[-1], oc['channel1']],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     trainable=trainable,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            o = tf.nn.conv2d(o, filter=weight, strides=[1, oc['stride'], oc['stride'],1], padding='SAME', name='conv2d-11-2')
            o = slim.batch_norm(o, is_training=trainable, fused=True, scope='batch-norm-11-2')
            o = tf.nn.relu(o, name='relu-11-2')

            weight = tf.get_variable('filter-256d-3',
                                     shape=[1, 1,  o.get_shape()[-1], oc['channel2']],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     trainable=trainable,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            o = tf.nn.conv2d(o, filter=weight, strides=[1, 1, 1,1], padding='SAME', name='conv2d-256d-3')
            o = slim.batch_norm(o, is_training=trainable, fused=True, scope='batch-norm-13')

            _, ow, oh, oc = list(o.get_shape())
            if oc != ic:
                ox = x
                if ow != iw:
                    ox = slim.conv2d(x, ic, [3, 3], 2, padding='SAME', reuse=reuse, scope='add-dimensions-adjust')
                ox = slim.conv2d(ox, oc, [1, 1], 1, padding='SAME', reuse=reuse, scope='add-dimensions')
                o = tf.add(ox, o)
            else:
                o = tf.add(x, o)
            o = tf.nn.relu(o, name='relu-256d-3')
        return o

