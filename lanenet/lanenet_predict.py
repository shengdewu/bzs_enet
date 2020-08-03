from lanenet.lanenet_model import lanenet_model
import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
import os
import traceback
import numpy as np
import cv2
import lanenet.img_queue
import time
import pandas as pd
from sklearn.cluster import MeanShift

class lanenet_predict(object):
    def __init__(self):
        self.delta_v = 0.5
        self.delta_d = 3.0
        return

    def infer(self, config):
        logging.info('ready for infer, param={}'.format(config))
        print('ready for infer, param={}'.format(config))
        lannet_net = lanenet_model()
        img_queue = lanenet.img_queue.img_queue(config['image_path'] + '/test_files.txt')
        # img_batch = img_queue.next_batch(config['eval_batch_size'],config['img_width'], config['img_height'])
        # cv2.imshow("resize", img_batch[0])
        # cv2.waitKey()
        with tf.device(config['device']):

            lanenet_image = tf.placeholder(tf.float32, shape=[None, config['img_height'], config['img_width'], 3])

            binary_image_predict, pix_embedding_predict = lannet_net.build_net(lanenet_image, config['eval_batch_size'], config['l2_weight_decay'], skip=config['skip'], is_trainging=False)
            binary_image_predict = tf.argmax(slim.softmax(binary_image_predict), axis=-1)
            
        print('restore from {}'.format(config['mode_path']))
        logging.info('restore from {}'.format(config['mode_path']))
        restore = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=config['device_log'])) as sess:
            restore.restore(sess=sess, save_path=config['mode_path'])

            while img_queue.is_continue(config['eval_batch_size']):
                try:
                    start = time.time()
                    lanenet_batch = img_queue.next_batch(config['eval_batch_size'], config['img_width'], config['img_height'])
                    binary_image, pix_embedding = sess.run([binary_image_predict, pix_embedding_predict], feed_dict={lanenet_image: lanenet_batch})
                    print('predict {} cost {}/s'.format(img_queue.batch(), time.time()-start))
                    self.post_processing(img_queue.batch(), config['result_path'], lanenet_batch, binary_image, pix_embedding)
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

    def cluster(self, binary_image, pix_embedding, max_center=4):
        embedding_value, coordinate = self.get_lanenet(binary_image, pix_embedding)
        cluster = MeanShift(bandwidth=1.0)
        cluster.fit(embedding_value)
        labels = cluster.labels_
        center = cluster.cluster_centers_

        kmeans = center.shape[0]

        cluster_index = list(np.unique(labels))
        if kmeans > max_center:
            cluster_index.clear()
            label_cnt = dict()
            for index in range(kmeans):
                idx = np.where(labels == index)
                label_cnt[index] = len(idx[0])
            sort_label_cnt = sorted(label_cnt.items(), key=lambda a:a[1], reverse=True)
            for slc in sort_label_cnt[0: max_center]:
                cluster_index.append(slc[0])

        cluster_coordinate = list()
        for c in cluster_index:
            idx = np.where(labels == c)[0]
            cluster_coordinate.append(coordinate[idx])
        return cluster_coordinate

    def post_processing(self, indice, out_path, src_imgs, binary_imgs, pix_embeddings, max_center=6):

        if out_path == '':
            return

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        save_path = out_path + '/' + str(indice)

        color = int(255 / max_center)
        for batch in range(np.shape(src_imgs)[0]):
            binary = binary_imgs[batch]
            embedding = pix_embeddings[batch]
            image = src_imgs[batch]
            #cluster_coordinate = self.cluster(binary, embedding, max_center)
            img_shape = image.shape
            # mask = np.zeros(shape=(img_shape[0], img_shape[1]))
            # for i, cc in enumerate(cluster_coordinate):
            #     for c in cc:
            #         mask[c[0], c[1]] = (i+1) * color

            #cv2.imwrite(save_path + '-' + str(indice) + '-mask.png', mask)
            cv2.imwrite(save_path + '-' + str(indice) + '-image.png', image)
            cv2.imwrite(save_path + '-' + str(indice) + '-binary-predict.png', binary * 255)
            feature_dim = np.shape(embedding)[-1]
            for i in range(feature_dim):
                embedding[:,:,i] = self.minmax_scale(embedding[:,:,i])
            cv2.imwrite(save_path + '-' + str(indice) + '-embedding-predict.png', embedding)
        return

    def get_lanenet(self, binary_img, pix_embedding):
        idx = np.where(binary_img == 1)
        coordinate = list()
        embedding = list()
        for i in range(len(idx[0])):
            embedding.append(pix_embedding[idx[0][i], idx[1][i]])
            coordinate.append((idx[0][i], idx[1][i]))
        return np.array(embedding, np.float), np.array(coordinate, np.int)



