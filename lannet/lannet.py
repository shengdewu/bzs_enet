from enet.enet import enet
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from config.config import config
import weight.weight
from enet.nn import nn
import time

class lannet(object):
    def __init__(self):
        self._back_bone = enet()
        return

    def load_image(self, image_path, image_type='train'):
        image_files = sorted([str('{}/{}/{}').format(image_path, image_type, file) for file in os.listdir(str('{}/{}').format(image_path, image_type)) if file.endswith('.png')])
        annot_image_files = sorted([str('{}/{}annot/{}').format(image_path, image_type, file) for file in os.listdir(str('{}/{}annot').format(image_path, image_type)) if file.endswith('.png')])
        return image_files, annot_image_files

    def preprocess(self, image, height=360, width=480, ch=3, dtype=tf.float32):
        if image.dtype != dtype:
            image = tf.image.convert_image_dtype(image, dtype=dtype)
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)
        image.set_shape((height, width, ch))
        return image

    def construct_image_tensor(self, image_path, batch_size, class_num, img_type='train'):
        image_files, image_annot_files = self.load_image(image_path, img_type)
        image_tensor = tf.convert_to_tensor(image_files)
        image_annot_tensor = tf.convert_to_tensor(image_annot_files)
        image_queue = tf.train.slice_input_producer([image_tensor, image_annot_tensor])
        pre_images = self.preprocess(tf.image.decode_image(tf.read_file(image_queue[0])), ch=3)
        pre_images_annot = self.preprocess(tf.image.decode_image(tf.read_file(image_queue[1])), ch=1, dtype=tf.uint8)

        images, images_annot = tf.train.batch([pre_images, pre_images_annot], batch_size=batch_size, allow_smaller_final_batch=True)

        image_annot_shape = images_annot.get_shape().as_list()
        images_annot = tf.reshape(images_annot, [batch_size, image_annot_shape[1], image_annot_shape[2]])
        images_onehot = tf.one_hot(images_annot, class_num)

        step_num_per_epoch = int(len(image_files) / batch_size)

        return images, images_annot, images_onehot, step_num_per_epoch

    def create_metrics(self, probabilities, images_annot, class_num):
        predict = tf.argmax(probabilities, axis=-1)
        accuracy, acc_update_op = tf.metrics.accuracy(labels=images_annot, predictions=predict)
        mean_iou, iou_update_op = tf.metrics.mean_iou(labels=images_annot, predictions=predict, num_classes=class_num)
        metrics_op = tf.group(acc_update_op, iou_update_op)
        return accuracy, mean_iou, metrics_op

    def get_weight(self, image_path, class_num):
        image_files, image_annot_files = self.load_image(image_path)
        w = weight.weight.median_frequency_balancing(image_annot_files, class_num)
        return w

    def train(self, config_path):
        network_config = config.get_config(config_path)

        images, images_annot, images_onehot, step_num_per_epoch = self.construct_image_tensor(network_config['image_path'],
                                                                                             network_config['batch_size'],
                                                                                             network_config['class_num'])

        with slim.arg_scope(nn.enet_arg_scope(weight_decay=network_config['l2_weight_decay'])):
            logits, probabilities = self._back_bone.building_net(input=images,
                                                                 batch_size=network_config['batch_size'],
                                                                 c=network_config['class_num'],
                                                                 stage_two_three=network_config['stage_two_three'],
                                                                 repeat_init_block=network_config['repeat_init_block'],
                                                                 skip=network_config['skip'],
                                                                 reuse=None)

        w = self.get_weight(network_config['image_path'], network_config['class_num'])
        w = images_onehot * w
        w = tf.reduce_sum(w, axis=3)
        losses = tf.losses.softmax_cross_entropy(onehot_labels=images_onehot, logits=logits, weights=w)
        total_loss = tf.losses.get_total_loss()

        accuracy, mean_iou, metrics_op = self.create_metrics(probabilities, images_annot, network_config['class_num'])

        global_step = tf.train.get_or_create_global_step()
        decay_steps = network_config['num_epochs_before_decay'] * step_num_per_epoch

        learning_rate_dec = tf.train.exponential_decay(learning_rate=network_config['learning_rate'], global_step=global_step, decay_steps=decay_steps, decay_rate=network_config['decay_rate'], staircase=True)
        train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_dec, epsilon=network_config['epsilon'])
        train_op = slim.learning.create_train_op(total_loss, train_optimizer)

        val_images, val_images_annot,val_images_onehot, val_step_num_per_epoch = self.construct_image_tensor(network_config['image_path'],
                                                                                                            network_config['eval_batch_size'],
                                                                                                            network_config['class_num'],
                                                                                                            'val')

        with slim.arg_scope(nn.enet_arg_scope(weight_decay=network_config['l2_weight_decay'])):
            val_logits, val_probabilities = self._back_bone.building_net(input=val_images,
                                                                        batch_size=network_config['eval_batch_size'],
                                                                        c=network_config['class_num'],
                                                                        stage_two_three=network_config['stage_two_three'],
                                                                        repeat_init_block=network_config['repeat_init_block'],
                                                                        skip=network_config['skip'],
                                                                        reuse=True)

        val_accuracy, val_mean_iou, val_metrics_op = self.create_metrics(val_probabilities, val_images_annot, network_config['class_num'])

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                for step in range(step_num_per_epoch * network_config['num_epoch']):

                    start_time = time.time()
                    loss, global_step_cnt, acc, iou, update_op = sess.run([train_op, global_step, accuracy, mean_iou, metrics_op])
                    print('train epoch:{}({}s)-loss={},acc={},iou={}'.format(global_step_cnt, time.time()-start_time, loss, acc, iou))

                    if step % min(network_config['batch_size'], step_num_per_epoch) == 0:
                        for i in range(val_step_num_per_epoch):
                            start_time = time.time()
                            _, acc, iou = sess.run([val_metrics_op, val_accuracy, val_mean_iou])
                            print('val epoch:{}({}s)-acc={},iou={}'.format(step, time.time()-start_time, acc, iou))
                    if step % network_config['update_mode_freq'] == 0:
                        print('save sess to {}'.format(network_config['mode_path']))
                        saver.save(sess, network_config['mode_path'])
            except Exception as err:
                print('{}'.format(err))
            finally:
                coord.request_stop()
                coord.join(threads)
            coord.join(threads)

        print('test')
        return
