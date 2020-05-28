from lannet.lanenet_model import lanenet_model
import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
import os
import traceback
import numpy as np
import cv2
import lannet.img_queue

class lanenet_predict(object):
    def __init__(self):
        self.delta_v = 0.5
        self.delta_d = 3.0
        return

    def infer(self, config):
        logging.info('ready for infer, param={}'.format(config))
        print('ready for infer, param={}'.format(config))
        lannet_net = lanenet_model()
        img_queue = lannet.img_queue.img_queue(config['image_path'] + '/test_files.txt')
        # img_batch = img_queue.next_batch(config['eval_batch_size'],config['img_width'], config['img_height'])
        # cv2.imshow("resize", img_batch[0])
        # cv2.waitKey()
        with tf.device(config['device']):

            lanenet_image = tf.placeholder(tf.float32, shape=[None, config['img_height'], config['img_width'], 3])

            binary_image_predict, pix_embedding_predict = lannet_net.build_net(lanenet_image, config['eval_batch_size'], config['l2_weight_decay'], skip=config['skip'])
            binary_image_predict = tf.argmax(slim.softmax(binary_image_predict), axis=-1)
            
        print('restore from {}'.format(config['mode_path']))
        logging.info('restore from {}'.format(config['mode_path']))
        restore = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=config['device_log'])) as sess:
            restore.restore(sess=sess, save_path=config['mode_path'])

            while img_queue.is_continue(config['eval_batch_size']):
                print('predict {}'.format(img_queue.batch()))
                try:
                    lanenet_batch = img_queue.next_batch(config['eval_batch_size'],config['img_width'], config['img_height'])
                    binary_image, pix_embedding = sess.run([binary_image_predict, pix_embedding_predict], feed_dict={lanenet_image: lanenet_batch})
                    self.save_image(img_queue.batch(), config['result_path'], lanenet_batch, binary_image, pix_embedding)
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

    def save_image(self, indice, out_path, src_images, binary_images, pix_embedding):
        if out_path == '':
            return

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        save_path = out_path + '/' + str(indice)

        for index in range(np.shape(src_images)[0]):
            binary = binary_images[index]
            embedding = pix_embedding[index]
            image = src_images[index]

            cv2.imwrite(save_path + '-' + str(index) + '-src-image.png', image)
            cv2.imwrite(save_path + '-' + str(index) + '-binary-predict.png', binary*255)
            feature_dim = np.shape(embedding)[-1]
            for i in range(feature_dim):
                embedding[:,:,i] = self.minmax_scale(embedding[:,:,i])

            cv2.imwrite(save_path + '-' + str(index) + '-embedding-predict.png', embedding)

            # cv2.imshow('src', image)
            # cv2.imshow('binary', (binary*255).astype(np.int8))
            # cv2.imshow('embedding', embedding.astype(np.int8))
            # cv2.waitKey()
        return

