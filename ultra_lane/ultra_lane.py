from resnet.resnet import resnet
import tensorflow as tf
import tensorflow.contrib.slim as slim
import util.data_pipe
from ultra_lane.data_stream import data_stream
import tusimple_process.ultranet_comm
import cv2
from ultra_lane.similarity_loss import similaryit_loss
from ultra_lane.similarity_loss import structural_loss

class ultra_lane():
    def __init__(self):
        #height = 720
        self._row_anchors = tusimple_process.ultranet_comm.ROW_ANCHORS
        self._cells = tusimple_process.ultranet_comm.CELLS
        self._lanes = tusimple_process.ultranet_comm.LANES
        return

    def make_net(self, x, label, trainable=True, reuse=False):
        b, w, h, c = label.get_shape().as_list()

        resnet_model = resnet()
        resnet_model.resnet18(x, self._cells+1, trainable, reuse)
        x3 = resnet_model.layer3
        x4 = resnet_model.layer4
        x5 = resnet_model.layer5

        total_dims = (self._cells + 1) * len(self._row_anchors) * self._lanes
        fc = slim.conv2d(x5, 8, [1, 1], 1, padding='SAME', reuse=reuse, scope='fc-1')
        fc = tf.reshape(fc, shape=(-1, 1800))
        fc = tf.contrib.layers.fully_connected(fc, 2048, scope='line1')
        fc = tf.contrib.layers.fully_connected(fc, total_dims, scope='line2')
        group_cls = tf.reshape(fc, shape=(-1, len(self._row_anchors), self._lanes, self._cells+1))

        label = tf.reshape(label, (b, w, h))
        label_oh = tf.one_hot(label, self._cells+1)
        cls = tf.losses.softmax_cross_entropy(label_oh, group_cls)

        sim = similaryit_loss(group_cls)

        shp = structural_loss(group_cls)

        return cls, sim, shp

    def train(self, config):
        data_handle = data_stream(config['image_path'], config['img_width'], config['img_height'])
        pipe_handle = util.data_pipe.data_pipe()
        with tf.device(config['device']):
            src_tensor, cls_tensor = data_handle.create_img_tensor()
            src_img_queue, cls_label_queue = pipe_handle.make_pipe(config['batch_size'], (src_tensor, cls_tensor), data_handle.pre_process_img)
            cls_loss_tensor, sim_loss_tensor, shp_loss_tensor = self.make_net(src_img_queue, cls_label_queue)
            total_loss_tensor = cls_loss_tensor + sim_loss_tensor + shp_loss_tensor
            with tf.Session(config=tf.ConfigProto(log_device_placement=config['device_log'])) as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                total_loss = sess.run(total_loss_tensor)
                print(total_loss)
        return

