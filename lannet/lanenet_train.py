from lannet.lanenet_model import lanenet_model
import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
import os
from weight import weight
from losses import discriminative
import traceback
import sys
import time
import matplotlib.pyplot as plt
from lannet.evaluate import lanenet_evalute
import numpy as np
import cv2

class lanenet_train(object):
    def __init__(self):
        self.delta_v = 0.5
        self.delta_d = 3.0
        return

    def _featch_img_paths(self, img_path):
        binary_img_files = list()
        instance_img_files = list()
        src_img_files = list()
        with open(img_path, 'r') as handler:
            while True:
                line = handler.readline()
                if not line:
                    break
                path = line.strip('\n')
                pathes = path.split(' ')
                for p in pathes:
                    if not os.path.exists(p):
                        logging.info('{} is not exists'.format(path))
                        raise FileExistsError('{} is not exists'.format(path))

                binary_img_files.append(pathes[0])
                instance_img_files.append(pathes[1])
                src_img_files.append(pathes[2])
        return binary_img_files, instance_img_files, src_img_files

    def _construct_img_queue(self, root_path, batch_size, width, height, sub_path='train_files.txt'):
        img_path = root_path + '/' + sub_path
        binary_img_files, instance_img_files, src_img_files = self._featch_img_paths(img_path)

        binary_img_tensor = tf.convert_to_tensor(binary_img_files)
        instance_img_tensor = tf.convert_to_tensor(instance_img_files)
        src_img_tensor = tf.convert_to_tensor(src_img_files)

        img_producer = tf.train.slice_input_producer([src_img_tensor, binary_img_tensor, instance_img_tensor])

        src_img = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(img_producer[0]), channels=3), height, width)
        binary_img = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(img_producer[1]), channels=1), height, width)
        instance_img = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(img_producer[2]), channels=1), height, width)

        binary_img = tf.divide(binary_img, tf.reduce_max(binary_img))
        binary_img = tf.cast(binary_img, tf.uint8)

        src_img = tf.cast(src_img, tf.float32)

        return tf.train.batch([src_img, binary_img, instance_img], batch_size=batch_size, allow_smaller_final_batch=True), len(binary_img_files)

    def train(self, config):
        logging.info('ready for training, param={}'.format(config))
        print('ready for training, param={}'.format(config))
        lannet_net = lanenet_model()
        with tf.device(config['device']):
            #train
            [src_queue, binary_queue, instance_queue], total_files = self._construct_img_queue(config['image_path'], config['batch_size'], config['img_width'], config['img_height'])
            binary_logits, embedding_logits = lannet_net.build_net(src_queue, config['batch_size'], config['l2_weight_decay'], skip=config['skip'])
            binary_acc = lanenet_evalute.accuracy(binary_queue, binary_logits)
            binary_fn = lanenet_evalute.fn(binary_queue, binary_logits)

            feature_dim = instance_queue.get_shape().as_list()
            instance_queue = tf.reshape(instance_queue, [config['batch_size'], feature_dim[1], feature_dim[2]])
            embedding_loss, l_var, l_dist, l_reg = discriminative.discriminative_loss_batch(prediction=embedding_logits, correct_label=instance_queue,
                                                                                            feature_dim=embedding_logits.get_shape().as_list()[3], image_shape=(feature_dim[1], feature_dim[2]),
                                                                                            delta_v=self.delta_v, delta_d=self.delta_d, param_var=1.0, param_dist=1.0, param_reg=0.001)
            l2_reg_loss = tf.losses.get_regularization_loss()

            binary_loss = self.caculate_binary_loss(binary_queue, binary_logits, config['batch_size'])

            total_loss = l2_reg_loss + binary_loss + embedding_loss

            global_setp = tf.train.create_global_step()
            exponential_decay_learning = tf.train.polynomial_decay(learning_rate=config['learning_rate'],
                                                                   global_step=global_setp,
                                                                   decay_steps=config['num_epochs_before_decay'],
                                                                   power=config['decay_rate'])

            optimizer = tf.train.MomentumOptimizer(learning_rate=exponential_decay_learning, momentum=config['epsilon'])
            train_op = slim.learning.create_train_op(total_loss, optimizer)

            total_loss_summary = tf.summary.scalar(name='total-loss', tensor=total_loss)
            train_summary_op = tf.summary.merge([total_loss_summary])

            #valid
            [test_src_queue, test_binary_queue, test_instance_queue], total_files = self._construct_img_queue(config['image_path'], config['eval_batch_size'], config['img_width'], config['img_height'], 'test_files.txt')

            test_binary_logits, test_embedding_logits = lannet_net.build_net(test_src_queue, config['eval_batch_size'], config['l2_weight_decay'], skip=config['skip'], reuse=True)

            test_feature_dim = test_instance_queue.get_shape().as_list()
            test_instance_label = tf.reshape(test_instance_queue, [config['eval_batch_size'], test_feature_dim[1], test_feature_dim[2]])
            test_embedding_loss, _, _, _ = discriminative.discriminative_loss_batch(prediction=test_embedding_logits, correct_label=test_instance_label,
                                                                                    feature_dim=test_embedding_logits.get_shape().as_list()[3], image_shape=(test_feature_dim[1], test_feature_dim[2]),
                                                                                    delta_v=self.delta_v, delta_d=self.delta_d, param_var=1.0, param_dist=1.0, param_reg=0.001)
            test_binary_loss = self.caculate_binary_loss(test_binary_queue, test_binary_logits, config['eval_batch_size'])

            test_binary_predict = tf.argmax(slim.softmax(test_binary_logits), axis=-1)

            input_shape = test_binary_queue.get_shape().as_list()
            test_binary_label = tf.reshape(test_binary_queue, shape=[config['eval_batch_size'], input_shape[1], input_shape[2]])

        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=config['device_log'])) as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            summary_writer = tf.summary.FileWriter(config['result_path']+'/summary')
            summary_writer.add_graph(sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                min_loss = sys.float_info.max
                for step in range(config['train_epoch']):
                    start_time = time.time()
                    loss, b_loss, e_loss, lg_loss, var, dist, reg, acc, fn, learning_rate,train_summary = sess.run([train_op, binary_loss, embedding_loss, l2_reg_loss,l_var, l_dist, l_reg, binary_acc,binary_fn,exponential_decay_learning,train_summary_op])
                    print('train epoch:{}({}s)-total_loss={},embedding_loss={},binary_loss={}, leg_loss={}, binary_acc/binary_fn={},{}, learning_ratg/decay_learning={},{}'.format(step, time.time() - start_time, loss, e_loss, b_loss, lg_loss,acc, fn, config['learning_rate'],learning_rate))
                    logging.info('train:{}({}s)-total_loss={},embedding_loss={},binary_loss={}, leg_loss={}, binary_acc/binary_fn={},{}, learning_ratg/decay_learning={},{}'.format(step, time.time() - start_time, loss, e_loss, b_loss, lg_loss,acc, fn, config['learning_rate'],learning_rate))

                    summary_writer.add_summary(train_summary, global_step=step)

                    if step % config['update_mode_freq'] == 0 and min_loss > loss:
                        print('save sess to {}, loss from {} to {}'.format(config['mode_path'], min_loss, loss))
                        logging.info('save sess to {}, loss from {} to {}'.format(config['mode_path'], min_loss, loss))
                        min_loss = loss
                        saver.save(sess, config['mode_path'])

                        start_time = time.time()
                        val_embedding_loss, val_binary_loss = sess.run([test_embedding_loss, test_binary_loss])
                        print('val epoch:{}({}s)-embedding_loss={},binary_loss={}'.format(step, time.time()-start_time, val_embedding_loss, val_binary_loss))
                        logging.info('val epoch:{}({}s)-embedding_loss={},binary_loss={}'.format(step, time.time()-start_time, val_embedding_loss, val_binary_loss))

                        test_images, test_binary_images,  test_embedding_images, test_binary_labels, test_instance_labels = sess.run([test_src_queue, test_binary_predict,  test_embedding_logits, test_binary_label, test_instance_label])
                        self.save_image(config['eval_batch_size'], config['result_path'], test_images, test_binary_labels, test_instance_labels, test_binary_images, test_embedding_images)

                print('train finish:min_loss={}'.format(min_loss))
                logging.info('train finish:min_loss={}'.format(min_loss))
            except Exception as err:
                print('{}'.format(err))
                logging.error('err:{}\n,track:{}'.format(err, traceback.format_exc()))
            finally:
                coord.request_stop()
                coord.join(threads)
            coord.join(threads)

        # print('restore from {}'.format(config['mode_path']))
        # logging.info('restore from {}'.format(config['mode_path']))
        # restore = tf.train.Saver()
        # with tf.Session(config=tf.ConfigProto(log_device_placement=config['device_log'])) as sess:
        #     restore.restore(sess=sess, save_path=config['mode_path'])
        #
        #     coord = tf.train.Coordinator()
        #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #     try:
        #         test_images, test_binary_images, test_embedding_images = sess.run([test_src_queue, test_binary_predict, test_embedding_predict])
        #         self.save_image(config['eval_batch_size'], config['result_path'], test_images, test_binary_images, test_embedding_images)
        #     except Exception as err:
        #         print('{}'.format(err))
        #         logging.error('err:{}\n,track:{}'.format(err, traceback.format_exc()))
        #     finally:
        #         coord.request_stop()
        #         coord.join(threads)
        #     coord.join(threads)
        return

    def caculate_binary_loss(self, binary_queue, binary_logits, batch_size):
        shape = binary_queue.get_shape().as_list()
        binary_queue = tf.reshape(binary_queue, [batch_size, shape[1], shape[2]])
        binary_onehot_queue = tf.one_hot(binary_queue, 2)

        w = weight.inverse_class_probability_weighting(binary_queue, 2)
        w = binary_onehot_queue * w
        w = tf.reduce_sum(w, axis=3)
        binayr_loss = tf.losses.softmax_cross_entropy(binary_onehot_queue, binary_logits, weights=w)
        return binayr_loss

    def minmax_scale(self, input_arr):
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        if max_val - min_val == 0:
            return input_arr

        return (input_arr - min_val) * 255.0 / (max_val - min_val)

    def save_image(self, max_index, out_path, src_images, binary_label_images, instance_label_images, binary_logits_images, embedding_logits_images):
        if out_path == '':
            return

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for index in range(max_index):
            binary = binary_logits_images[index]
            embedding = embedding_logits_images[index]
            binary_label = binary_label_images[index]
            embedding_label = instance_label_images[index]
            image = src_images[index]

            cv2.imwrite(out_path + '/' + str(index) + '-image.png', image)
            cv2.imwrite(out_path + '/' + str(index) + '-binary-predict.png', binary*255)
            feature_dim = np.shape(embedding)[-1]
            for i in range(feature_dim):
                embedding[:,:,i] = self.minmax_scale(embedding[:,:,i])
            cv2.imwrite(out_path + '/' + str(index) + '-embedding-predict.png', embedding)
            cv2.imwrite(out_path + '/' + str(index) + '-binary-label.png', binary_label*255)
            cv2.imwrite(out_path + '/' + str(index) + '-embedding-label.png', embedding_label)
        return
