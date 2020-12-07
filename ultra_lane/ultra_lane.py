from resnet.resnet import resnet
import tensorflow as tf
import tensorflow.contrib.slim as slim
import util.data_pipe
from ultra_lane.data_stream import data_stream
'''
the row anchors of tusimple dataset, in which the image height is 720, range from 160 to 710 with step of 10
the number of gridding cells is set to 100 on tusimple dataset     
'''
class ultra_lane():
    def __init__(self):
        #height = 720
        self._row_anchors = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
        self._cells = 100
        self._lanes = 4
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
        data_handle = data_stream(config['height'], config['width'], self._row_anchors, self._lanes, self._cells, config['root'])
        pipe_handle = util.data_pipe.data_pipe()

        src_img, cls_label = pipe_handle.make_pipe(config['batch_size'], data_handle.create_img_tensor, data_handle.pre_process_img)

        return

