from lanenet.lanenet_model import lanenet_model
import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
import os
import traceback
import numpy as np
import cv2
import time
from timg.timg import timg

class lanenet_realtime(object):
    def __init__(self, target):
        self.delta_v = 0.5
        self.delta_d = 3.0
        self.target = target
        return

    def create_output(self, path):
        if path == '':
            raise FileNotFoundError('the save path is not specified!!!')
        if not os.path.exists(path):
            os.makedirs(path)
        return

    def start(self, config):
        logging.info('ready for infer, param={}'.format(config))
        print('ready for infer, param={}'.format(config))
        self.create_output(config['result_path'])
        lannet_net = lanenet_model()
        with tf.device(config['device']):
            test_src_queue = tf.placeholder(dtype=tf.float32, shape=[1, config['img_height'], config['img_width'], 3])
            binary_image_predict, pix_embedding_predict = lannet_net.build_net(test_src_queue, 1, config['l2_weight_decay'], skip=config['skip'], is_trainging=False)
            binary_image_predict = tf.argmax(slim.softmax(binary_image_predict), axis=-1)

        print('restore from {}'.format(config['mode_path']))
        logging.info('restore from {}'.format(config['mode_path']))
        restore = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=config['device_log'])) as sess:
            restore.restore(sess=sess, save_path=config['mode_path'])
            while True:
                try:
                    pre_process = self.preprocess(config['img_width'], config['img_height'])
                    start = time.time()
                    src_image, binary_image, pix_embedding = sess.run([test_src_queue, binary_image_predict, pix_embedding_predict], feed_dict={test_src_queue: pre_process})
                    cost = time.time() - start
                    print('predict cost {}/s'.format(cost))
                    self.post_processing(start, config['result_path'], src_image, binary_image, pix_embedding)
                    cost = time.time() - start - cost
                    print('post process cost {}/s'.format(cost))
                except Exception as err:
                    print('{}'.format(err))
                    logging.error('err:{}\n,track:{}'.format(err, traceback.format_exc()))
                    break
        return

    def preprocess(self, width, height):
        frame = self.target()
        change_frame = np.zeros(np.shape(frame), dtype=frame.dtype)
        axis_0 = frame[:, :, 0]
        axis_1 = frame[:, :, 1]
        axis_2 = frame[:, :, 2]
        change_frame[:, :, 0] = axis_2
        change_frame[:, :, 1] = axis_1
        change_frame[:, :, 2] = axis_0
        pre_process = timg.crop_pad(change_frame, height, width)
        # cv2.imshow('source', change_frame)
        # cv2.imshow('crop', pre_process)
        # cv2.waitKey(1)
        return [pre_process.astype(np.float32)]

    def minmax_scale(self, input_arr):
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        if max_val - min_val == 0:
            return input_arr

        return (input_arr - min_val) * 255.0 / (max_val - min_val)

    def post_processing(self, indice, out_path, src_imgs, binary_imgs, pix_embeddings):

        save_path = out_path + '/' + str(indice)

        binary = binary_imgs[0] * 255
        embedding = pix_embeddings[0]
        image = src_imgs[0]

        cv2.imwrite(save_path + 'image.png', image)
        cv2.imwrite(save_path + 'binary-predict.png', binary*255)
        feature_dim = np.shape(embedding)[-1]
        for i in range(feature_dim):
            embedding[:, :, i] = self.minmax_scale(embedding[:, :, i])
        cv2.imwrite(save_path + 'embedding-predict.png', embedding)

        image_dim = self.convert_dim(image.astype(np.uint8), 3, 4)
        binary_dim = self.convert_dim(cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), 3, 4)
        embedding_dim = embedding.astype(np.uint8)
        cv2.imshow("detect", np.hstack([image_dim, binary_dim*255, embedding_dim]))
        cv2.waitKey(1)
        return

    def convert_dim(self, image, src_dim, update_dim):
        shape = image.shape
        max_dim_img = np.zeros((shape[0], shape[1], update_dim), np.uint8)
        for dim in range(src_dim):
            max_dim_img[:,:,dim] = image[:,:,dim]
        return max_dim_img



