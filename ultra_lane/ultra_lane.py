from resnet.resnet import resnet
import tensorflow as tf
import tensorflow.contrib.slim as slim

class ultra_lane():
    def __init__(self):
        return

    def make_net(self, x, cell, anchors, lanes, trainable=True, reuse=False):
        resnet_model = resnet()
        resnet_model.resnet18(x, cell+1, trainable, reuse)
        x3 = resnet_model.layer3
        x4 = resnet_model.layer4
        x5 = resnet_model.layer5

        total_dims = (cell+1)*anchors*lanes
        fc = slim.conv2d(x5, 8, [1, 1], 1, padding='SAME', reuse=reuse, scope='fc-1')
        fc = tf.reshape(fc, shape=(-1, 1800))
        fc = tf.contrib.layers.fully_connected(fc, 2048)
        fc = tf.contrib.layers.fully_connected(fc, total_dims)
        group_cls = tf.reshape(fc, shape=(-1, cell+1, anchors, lanes))

        return group_cls
