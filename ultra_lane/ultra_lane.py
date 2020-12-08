from resnet.resnet import resnet
import tensorflow as tf
import tensorflow.contrib.slim as slim
import util.data_pipe
from ultra_lane.data_stream import data_stream
import tusimple_process.ultranet_comm
import cv2

class ultra_lane():
    def __init__(self):
        #height = 720
        self._row_anchors = tusimple_process.ultranet_comm.ROW_ANCHORS
        self._cells = tusimple_process.ultranet_comm.CELLS
        self._lanes = tusimple_process.ultranet_comm.LANES
        return

    def make_net(self, x, label, cell, anchors, lanes, trainable=True, reuse=False):
        resnet_model = resnet()
        resnet_model.resnet18(x, cell+1, trainable, reuse)
        x3 = resnet_model.layer3
        x4 = resnet_model.layer4
        x5 = resnet_model.layer5

        total_dims = (cell+1)*anchors*lanes
        fc = slim.conv2d(x5, 8, [1, 1], 1, padding='SAME', reuse=reuse, scope='fc-1')
        fc = tf.reshape(fc, shape=(-1, 1800))
        fc = tf.contrib.layers.fully_connected(fc, 2048, scope='line1')
        fc = tf.contrib.layers.fully_connected(fc, total_dims, scope='line2')
        group_cls = tf.reshape(fc, shape=(-1, cell+1, anchors, lanes))

        label_oh = tf.one_hot(label, cell+1)
        cls = tf.losses.softmax_cross_entropy(label_oh, group_cls)

        return group_cls

    def train(self, config):
        data_handle = data_stream(config['image_path'])
        pipe_handle = util.data_pipe.data_pipe()
        with tf.device(config['device']):
            src_tensor, cls_tensor = data_handle.create_img_tensor()
            src_img_queue, cls_label_queue = pipe_handle.make_pipe(config['batch_size'], (src_tensor, cls_tensor), data_handle.pre_process_img)

            with tf.Session(config=tf.ConfigProto(log_device_placement=config['device_log'])) as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                src_img, cls_label = sess.run([src_img_queue, cls_label_queue])
                cv2.imshow('src_img', src_img[0])
                cv2.imshow('cls', cls_label[0])
                cv2.waitKey()
                cv2.destroyAllWindows()
                print('')
        return

