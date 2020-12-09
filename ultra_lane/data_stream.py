from util import data_pipe
import os
import logging
import tensorflow as tf
import numpy as np

class data_stream:
    def __init__(self, root, img_w, img_h, label_w=4, label_h=56, file_name='train_gt.txt'):
        self._root = root
        self._file_name = file_name
        self._img_w = img_w
        self._img_h = img_h
        self._label_w = label_w
        self._label_h = label_h
        return

    def create_img_tensor(self):
        label_img_files = list()
        src_img_files = list()
        with open(self._root+'/'+self._file_name, 'r') as handler:
            while True:
                line = handler.readline()
                if not line:
                    break
                names = line.strip('\n').split(' ')
                img_path = self._root + '/' + names[0]
                label_path = self._root + '/' + names[1]
                if not os.path.exists(img_path) or not os.path.exists(label_path):
                    logging.info('{}-{} is not exists'.format(img_path, label_path))
                    continue

                src_img_files.append(img_path)
                label_img_files.append(label_path)

        label_img_tensor = tf.convert_to_tensor(label_img_files)
        src_img_tensor = tf.convert_to_tensor(src_img_files)
        return src_img_tensor, label_img_tensor

    def pre_process_img(self, src_img_tensor, label_img_tensor):
        src_img = tf.image.decode_jpeg(tf.read_file(src_img_tensor), channels=3)
        label_img = tf.image.decode_jpeg(tf.read_file(label_img_tensor), channels=1)

        src_img = tf.image.resize_image_with_crop_or_pad(src_img, self._img_h, self._img_w)
        label_img = tf.image.resize_image_with_crop_or_pad(label_img, self._label_h, self._label_w)

        label_img = tf.cast(label_img, tf.uint8)
        src_img = tf.cast(src_img, tf.float32)

        return src_img, label_img
