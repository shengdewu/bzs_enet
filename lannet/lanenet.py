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

class lanenet(object):
    def __init__(self):
        self._binary_img_path = 'gt_binary_img'
        self._instance_img_path = 'gt_instance_img'
        self._src_img_path = 'gt_src_img'
        self.delta_v = 0.5
        self.delta_d = 3
        return

    def _featch_img_paths(self, img_path):
        return [img_path + f for f in os.listdir(img_path) if f.endswith('.jpg')]

    def _construct_img_queue(self, root_path, batch_size, width, height, sub_path='train'):
        img_path = root_path + '/' + sub_path
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

        binary_img = tf.image.convert_image_dtype(binary_img, tf.uint8)
        src_img = tf.image.convert_image_dtype(src_img, tf.float32)

        return tf.train.batch([src_img, binary_img, instance_img], batch_size=batch_size, allow_smaller_final_batch=True), len(binary_img_files)

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
        return recall

    def train(self, config):
        [src_queue, binary_queue, instance_queue], total_files = self._construct_img_queue(config['image_path'], config['batch_size'], 512, 256)

        lannet_net = lanenet_model()
        binary_logits, embedding_logits = lannet_net.build_net(src_queue, config['batch_size'], config['l2_weight_decay'])

        ls_loss = tf.losses.get_total_loss()

        feature_dim = instance_queue.get_shape().as_list()
        instance_queue = tf.reshape(instance_queue, [config['batch_size'], feature_dim[1], feature_dim[2]])
        embedding_loss, l_var, l_dist, l_reg = discriminative.discriminative_loss_batch(embedding_logits, instance_queue, embedding_logits.get_shape().as_list()[3], (feature_dim[1], feature_dim[2]), self.delta_v, self.delta_d, 1.0, 1.0, 0.001)

        shape = binary_queue.get_shape().as_list()
        binary_queue = tf.reshape(binary_queue, [config['batch_size'], shape[1], shape[2]])
        binary_onehot_queue = tf.one_hot(binary_queue, 2)
        w = weight.median_frequency_balancing(self._featch_img_paths(config['image_path'] + '/train/' + self._binary_img_path + '/'), 2)
        w = binary_onehot_queue * w
        w = tf.reduce_sum(w, axis=3)
        binayr_loss = tf.losses.softmax_cross_entropy(binary_onehot_queue, binary_logits, weights=w)

        total_loss = ls_loss + binayr_loss + embedding_loss

        global_setp = tf.train.create_global_step()
        steps_per_epoch = int(total_files / config['batch_size'])
        decay_steps = config['num_epochs_before_decay'] * steps_per_epoch
        exponential_decay_learning = tf.train.exponential_decay(learning_rate=config['learning_rate'],
                                                                global_step=global_setp,
                                                                decay_rate=config['decay_rate'],
                                                                decay_steps=decay_steps,
                                                                staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=exponential_decay_learning, epsilon=config['epsilon'])
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        #验证
        [test_src_queue, test_binary_queue, test_instance_queue], val_total = self._construct_img_queue(config['img_path'], config['eval_batch_size'], 512, 256, 'test')
        test_binary_logits, test_embedding_logits = lannet_net.build_net(test_src_queue, config['eval_batch_size'], config['l2_weight_decay'])
        test_embedding_loss, test_l_var, test_l_dist, test_l_reg = discriminative.discriminative_loss_batch(test_embedding_logits, test_instance_queue, feature_dim, (feature_dim[0], feature_dim[1]), self.delta_v, self.delta_d, 1.0, 1.0, 0.001)
        test_acc = self._accuracy(test_binary_queue, test_binary_logits)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                min_loss = sys.float_info.max
                for step in range(config['num_epoch'] * steps_per_epoch):
                    start_time = time.time()
                    loss, b_loss, e_loss, var, dist, reg = sess.run([train_op, binayr_loss, embedding_loss, l_var, l_dist, l_reg])
                    print('train epoch:{}({}s)-total_loss={},embedding_loss={},binary_loss={}'.format(step, time.time() - start_time, loss, e_loss, b_loss))
                    logging.info('train:{}({}s)-total_loss={},embedding_loss={},binary_loss={}'.format(step, time.time() - start_time, loss, e_loss, b_loss))

                    if step % max(config['update_mode_freq'], steps_per_epoch) == 0:
                            start_time = time.time()
                            test_e_loss, test_b_loss = sess.run([test_embedding_loss, test_acc])
                            print('val epoch:{}({}s)-embedding_loss={},binary_loss={}'.format(step, time.time()-start_time, test_e_loss, test_b_loss))
                            logging.info('val epoch:{}({}s)-acc={},iou={}'.format(step, time.time()-start_time, test_e_loss, test_b_loss))

                    if (step+1) % config['update_mode_freq'] == 0 and min_loss > loss:
                        print('save sess to {}, loss from {} to {}'.format(config['mode_path'], min_loss, loss))
                        logging.info('save sess to {}, loss from {} to {}'.format(config['mode_path'], min_loss, loss))
                        min_loss = loss
                        saver.save(sess, config['mode_path'])
            except Exception as err:
                print('{}'.format(err))
                logging.error('err:{}\n,track:{}'.format(err, traceback.format_exc()))
            finally:
                coord.request_stop()
                coord.join(threads)
            coord.join(threads)
        return


