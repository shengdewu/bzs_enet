from lannet.lanenet_model import lanenet_model
import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
import os
import traceback
import numpy as np
import cv2
import lannet.img_queue

class lanenet_train(object):
    def __init__(self):
        self.delta_v = 0.5
        self.delta_d = 3.0
        return

    def infer(self, config):
        logging.info('ready for infer, param={}'.format(config))
        print('ready for infer, param={}'.format(config))
        lannet_net = lanenet_model()
        img_queue = lannet.img_queue.img_queue(config['image_path'] + '/test_files.txt')

        with tf.device(config['device']):

            lanenet_image = tf.placeholder(tf.float32, shape=[None, config['img_width'], config['img_height'], 3])

            binary_image, pix_embedding_predict, _ = lannet_net.build_net(lanenet_image, config['eval_batch_size'], config['l2_weight_decay'], skip=config['skip'], is_trainging=False)
            binary_image_predict = tf.argmax(slim.softmax(binary_image), axis=-1)
            
        print('restore from {}'.format(config['mode_path']))
        logging.info('restore from {}'.format(config['mode_path']))
        restore = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=config['device_log'])) as sess:
            restore.restore(sess=sess, save_path=config['mode_path'])

            while not img_queue.is_empty():
                try:
                    binary_image, pix_embedding = sess.run([binary_image_predict, pix_embedding_predict], feed_dict={lanenet_image, img_queue.next_batch(config['eval_batch_size'],config['img_width'], config['img_height'])})
                    fuse_image = np.multiply(binary_image, pix_embedding)
                    self.save_image(config['result_path'], lanenet_image, binary_image, pix_embedding, fuse_image)
                except Exception as err:
                    print('{}'.format(err))
                    logging.error('err:{}\n,track:{}'.format(err, traceback.format_exc()))

        return

    def minmax_scale(self, input_arr):
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        if max_val - min_val == 0:
            return input_arr

        return (input_arr - min_val) * 255.0 / (max_val - min_val)

    def save_image(self, out_path, src_images, binary_images, pix_embedding, fuse_image):
        if out_path == '':
            return

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        cv2.imwrite(out_path + '/src-image.png', src_images)
        cv2.imwrite(out_path + '/binary-predict.png', binary_images*255)
        feature_dim = np.shape(pix_embedding)[-1]
        for i in range(feature_dim):
            pix_embedding[:,:,i] = self.minmax_scale(pix_embedding[:,:,i])

        cv2.imwrite(out_path + '/embedding-predict.png', pix_embedding)
        cv2.imwrite(out_path + '/fuse_image.png', fuse_image)
        return

