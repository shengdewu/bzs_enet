from lannet.lanenet_model import lanenet_model
import tensorflow as tf
import logging
import os
from weight import weight
from losses import discriminative
import numpy as np

class lanenet(object):
    def __init__(self):
        self._binary_img_path = 'gt_binary_img'
        self._instance_img_path = 'gt_instance_img'
        self._src_img_path = 'gt_src_img'
        return

    def _featch_img_paths(self, img_path):
        return [img_path + f for f in os.listdir(img_path) if f.endswith('.jpg')]


    def _construct_img_queue(self, img_path, batch_size, width, height):
        binary_img_files = self._featch_img_paths(img_path + '/' + self._binary_img_path)
        instance_img_files = self._featch_img_paths(img_path + '/' + self._instance_img_path)
        src_img_files = self._featch_img_paths(img_path + '/' + self._src_img_path)

        binary_img_tensor = tf.convert_to_tensor(binary_img_files)
        instance_img_tensor = tf.convert_to_tensor(instance_img_files)
        src_img_tensor = tf.convert_to_tensor(src_img_files)

        img_producer = tf.train.slice_input_producer([src_img_tensor, binary_img_tensor, instance_img_tensor])

        src_img = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(img_producer[0]), channels=3), height, width)
        binary_img = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(img_producer[1]), channels=1), height, width) / 255
        instance_img = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(img_producer[2]), channels=1), height, width)

        return tf.train.batch([src_img, binary_img, instance_img], batch_size=batch_size, allow_smaller_final_batch=True)

    def _accuracy(self, label, predict):
        logit = tf.nn.softmax(predict, axis=-1)
        logit = tf.argmax(logit, axis=-1)
        logit = tf.cast(logit, tf.float64)
        logit = tf.expand_dims(logit, axis=-1)

        idx = tf.where(tf.equal(label, 1))
        out_logit = tf.gather_nd(logit, idx)
        acc = tf.count_nonzero(out_logit)
        total = tf.shape(out_logit)
        recall = tf.divide(acc, tf.cast(total[0], tf.int64))
        return [recall, logit, out_logit, acc, total]

    def train(self, config):
        src_queue, binary_queue, instance_queue = self._construct_img_queue(config['img_path'], config['batch_size'], config['width'], config['height'])

        lannet_net = lanenet_model()
        binary_logits, embedding_logits = lannet_net.build_net(src_queue, config['batch_size'], config['l2_weight_decay'])

        binary_onehot_queue = tf.one_hot(binary_queue, 2)

        w = weight.median_frequency_balancing(self._featch_img_paths(config['img_path'] + '/' + self._binary_img_path), 2)
        w = binary_onehot_queue * w
        w = tf.reduce_sum(w, axis=3)

        ls_loss = tf.losses.get_total_loss()

        binayr_loss = tf.losses.sparse_softmax_cross_entropy(binary_onehot_queue, binary_logits, weights=w)

        feature_dim = embedding_logits.get_shape().as_list()

        embedding_loss, l_var, l_dist, l_reg = discriminative.discriminative_loss_batch(embedding_logits, instance_queue, feature_dim, (feature_dim[0], feature_dim[1]), 0.5, 3.0, 1.0, 1.0, 0.001)

        return


