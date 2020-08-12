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
from lanenet.test_data_pipe import test_data_pipe

class lanenet_predict(object):
    def __init__(self):
        self.delta_v = 0.5
        self.delta_d = 3.0
        return

    def infer(self, config):
        logging.info('ready for infer, param={}'.format(config))
        print('ready for infer, param={}'.format(config))
        lannet_net = lanenet_model()
        data_handle = test_data_pipe(width=config['img_width'], height=config['img_height'])

        with tf.device(config['device']):
            test_src_queue = data_handle.make_pipe(config['image_path'], config['eval_batch_size'], 'test_files.txt')
            binary_image_predict, pix_embedding_predict = lannet_net.build_net(test_src_queue, config['eval_batch_size'], config['l2_weight_decay'], skip=config['skip'], is_trainging=False)
            binary_image_predict = tf.argmax(slim.softmax(binary_image_predict), axis=-1)
            
        print('restore from {}'.format(config['mode_path']))
        logging.info('restore from {}'.format(config['mode_path']))
        restore = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=config['device_log'])) as sess:
            restore.restore(sess=sess, save_path=config['mode_path'])

            try:
                ilter = 0
                while True:
                    start = time.time()
                    src_image, binary_image, pix_embedding = sess.run([test_src_queue, binary_image_predict, pix_embedding_predict])
                    cost = time.time()-start
                    print('predict {}/per, cost {}/s, mean cost {}/s'.format(config['eval_batch_size'], cost, cost/config['eval_batch_size']))
                    self.post_processing('iter-{}-b-{}'.format(ilter, config['eval_batch_size']), config['result_path'], src_image, binary_image, pix_embedding)
                    ilter += 1
            except tf.errors.OutOfRangeError as over:
                print('predict success\n')
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
            cv2.imwrite(save_path + '-' + str(batch) + '-image.png', image)
            cv2.imwrite(save_path + '-' + str(batch) + '-binary-predict.png', binary * 255)
            feature_dim = np.shape(embedding)[-1]
            for i in range(feature_dim):
                embedding[:,:,i] = self.minmax_scale(embedding[:,:,i])
            cv2.imwrite(save_path + '-' + str(batch) + '-embedding-predict.png', embedding)
        return

    def get_lanenet(self, binary_img, pix_embedding):
        idx = np.where(binary_img == 1)
        coordinate = list()
        embedding = list()
        for i in range(len(idx[0])):
            embedding.append(pix_embedding[idx[0][i], idx[1][i]])
            coordinate.append((idx[0][i], idx[1][i]))
        return np.array(embedding, np.float), np.array(coordinate, np.int)



