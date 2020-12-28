from resnet.resnet import resnet
import tensorflow as tf
import tensorflow.contrib.slim as slim
import util.data_pipe
from ultra_lane.data_stream import data_stream
import tusimple_process.ultranet_comm
import cv2
from ultra_lane.similarity_loss import similaryit_loss
from ultra_lane.similarity_loss import structural_loss
import logging
import numpy as np
from tusimple_process.create_label import tusimple_label
import os

class ultra_lane():
    def __init__(self):
        #height = 720
        self._row_anchors = tusimple_process.ultranet_comm.ROW_ANCHORS
        self._cells = tusimple_process.ultranet_comm.CELLS
        self._lanes = tusimple_process.ultranet_comm.LANES
        self.cls_label_handle = tusimple_label()
        return

    def make_net(self, x, trainable=True, reuse=False):
        resnet_model = resnet()
        resnet_model.resnet18(x, self._cells+1, trainable, reuse)
        x3 = resnet_model.layer3
        x4 = resnet_model.layer4
        x5 = resnet_model.layer5

        total_dims = (self._cells + 1) * len(self._row_anchors) * self._lanes
        fc = slim.conv2d(x5, 8, [1, 1], 1, padding='SAME', reuse=reuse, scope='fc-1')
        fc = tf.reshape(fc, shape=(-1, 1800))
        fc = tf.contrib.layers.fully_connected(fc, 2048, scope='line1', reuse=reuse, activation_fn=tf.nn.relu)
        fc = tf.contrib.layers.fully_connected(fc, total_dims, scope='line2', reuse=reuse, activation_fn=None)
        group_cls = tf.reshape(fc, shape=(-1, len(self._row_anchors), self._lanes, self._cells+1))

        return group_cls

    def loss(self, group_cls, label):
        bs, ws, hs, cs = label.get_shape().as_list()

        scores = tf.nn.softmax(group_cls, axis=3)
        factor = tf.pow(1.-scores, 2)
        log_score = tf.nn.log_softmax(group_cls, axis=3)
        log_score = factor * log_score

        label = tf.reshape(label, (bs, ws, hs))
        label_oh = tf.one_hot(label, self._cells+1)
        nllloss1 = tf.multiply(label_oh, log_score)
        nllloss2 = tf.abs(nllloss1)
        index = tf.where(nllloss2 > 0)
        nllloss3 = tf.gather_nd(nllloss2, index)

        cls = tf.reduce_mean(nllloss3)

        sim = similaryit_loss(group_cls)

        shp = structural_loss(group_cls)

        return cls, sim, shp, tf.argmax(slim.softmax(group_cls), axis=-1)

    def train(self, config):
        pipe_handle = util.data_pipe.data_pipe()
        save_path = config['result_path'] + '/out_img'
        os.makedirs(save_path, exist_ok=True)
        with tf.device(config['device']):
            #train
            train_data_handle = data_stream(config['image_path'], config['img_width'], config['img_height'])
            src_tensor, label_tensor, cls_tensor = train_data_handle.create_img_tensor()
            #train_data_handle.pre_process_img(src_tensor[0], label_tensor[0], cls_tensor[0])
            src_img_queue, label_queue, cls_queue = pipe_handle.make_pipe(config['batch_size'], (src_tensor, label_tensor, cls_tensor), train_data_handle.pre_process_img)
            group_cls = self.make_net(src_img_queue)
            cls_loss_tensor, sim_loss_tensor, shp_loss_tensor, lane_row_anchors_tensor = self.loss(group_cls, cls_queue)
            total_loss_tensor = cls_loss_tensor + sim_loss_tensor + shp_loss_tensor
            total_loss_summary = tf.summary.scalar(name='total-loss', tensor=total_loss_tensor)
            cls_loss_summary = tf.summary.scalar(name='cls-loss', tensor=cls_loss_tensor)

            b, w, h, c = cls_queue.get_shape().as_list()
            predict_label = tf.cast(tf.reshape(cls_queue, (b, w, h)), dtype=lane_row_anchors_tensor.dtype)
            correct_label = tf.cast(tf.equal(predict_label, lane_row_anchors_tensor), dtype=tf.float32)
            precision = tf.reduce_sum(correct_label) / (1.0*w*h*b*c)

            global_step = tf.train.create_global_step()
            learning_rate = tf.train.exponential_decay(config['learning_rate'], global_step, config['num_epochs_before_decay'], config['decay_rate'])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=config['epsilon'])
            train_op = slim.learning.create_train_op(total_loss_tensor, optimizer)
            ls_summary = tf.summary.scalar(name='learning-rate', tensor=learning_rate)

            #valid
            valid_data_handle = data_stream(config['image_path'], config['img_width'], config['img_height'], 'valid_files.txt')
            valid_src_tensor, valid_label_tensor, valid_cls_tensor = valid_data_handle.create_img_tensor()
            valid_src_img_queue, valid_label_queue, valid_cls_queue = pipe_handle.make_pipe(config['batch_size'], (valid_src_tensor, valid_label_tensor, valid_cls_tensor), valid_data_handle.pre_process_img)
            valid_group_cls = self.make_net(valid_src_img_queue, False, True)
            valid_cls_loss_tensor, valid_sim_loss_tensor, valid_shp_loss_tensor, valid_lane_row_anchors_tensor = self.loss(valid_group_cls, valid_cls_queue)
            valid_total_loss_tensor = valid_cls_loss_tensor + valid_sim_loss_tensor + valid_shp_loss_tensor
            val_total_loss_summary = tf.summary.scalar(name='val-total-loss', tensor=valid_total_loss_tensor)
            val_cls_loss_summary = tf.summary.scalar(name='val-cls-loss', tensor=valid_cls_loss_tensor)

            train_summary_op = tf.summary.merge([total_loss_summary, cls_loss_summary, ls_summary, val_total_loss_summary, val_cls_loss_summary])

            #saver = tf.train.Saver()
            with tf.Session(config=tf.ConfigProto(log_device_placement=config['device_log'])) as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                summary_writer = tf.summary.FileWriter(config['result_path'] + '/summary')
                summary_writer.add_graph(sess.graph)

                min_loss = float('inf')
                for step in range(config['train_epoch']):

                    _, cls_loss, sim_loss, shp_loss, train_summary, gs, lr, lane_img, label_row_anchor, train_row_anchor, p = sess.run([train_op, cls_loss_tensor, sim_loss_tensor, shp_loss_tensor, train_summary_op, global_step, learning_rate, src_img_queue, cls_queue, lane_row_anchors_tensor, precision])

                    total_loss = cls_loss + sim_loss + shp_loss

                    summary_writer.add_summary(train_summary, global_step=gs)

                    logging.info('train model: gs={},  loss={},[{},{},{}], precision={}, lr={}'.format(gs, total_loss, cls_loss, sim_loss, shp_loss, p, lr))

                    if (step + 1) % config['update_mode_freq'] == 0:
                        val_total_loss, val_lane_img, val_label_row_anchor, val_row_anchor = sess.run([valid_total_loss_tensor, valid_src_img_queue, valid_cls_queue, valid_lane_row_anchors_tensor])
                        self.match_coordinate(val_lane_img.astype(np.uint8), val_label_row_anchor, val_row_anchor, save_path, step)
                        logging.info('valid model: gs={},  loss={}, lr={}'.format(gs, val_total_loss, lr))
                        print('valid model: gs={},  loss={}, lr={}'.format(gs, val_total_loss, lr))
                        print('train model: gs={},  loss={},[{},{},{}], precision={}, lr={}'.format(gs, total_loss, cls_loss, sim_loss, shp_loss, p, lr))
                        # if min_loss > total_loss:
                        #     saver.save(sess, config['mode_path'])
                        #     logging.info('update model loss from {} to {}'.format(min_loss, total_loss))

                    min_loss = min(min_loss, total_loss)
        return

    def match_coordinate(self, imgs, label_rows, train_rows, save_path, epoch):
        batch, h, w, c = imgs.shape

        for b in range(batch):
            label_lane = self.cls_label_handle.rescontruct(label_rows[b][:, :, 0], imgs[b].copy())
            predict_lane = self.cls_label_handle.rescontruct(train_rows[b], imgs[b].copy())
            all_img = np.hstack([label_lane, predict_lane])
            cv2.imwrite(save_path+'/'+str(epoch)+'-'+str(b)+'.png', all_img)
        return

