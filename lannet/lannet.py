from enet.enet import enet
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from config.config import config
import weight.weight

class lannet(object):
    def __init__(self):
        self._back_bone = enet()
        return

    def load_image(self, image_path, image_type='train'):
        image_files = sorted([str('{}/{}/{}').format(image_path, image_type, file) for file in os.listdir(str('{}/{}').format(image_path, image_type)) if file.endswith('.png')])
        annot_image_files = sorted([str('{}/{}annot/{}').format(image_path, image_type, file)  for file in os.listdir(str('{}/{}annot').format(image_path, image_type)) if file.endswith('.png')])
        return image_files, annot_image_files

    def preprocess(self, image, height=360, width=480, ch=3, dtype=tf.float32):
        if image.dtype != dtype:
            image = tf.image.convert_image_dtype(image, dtype=dtype)
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)
        image.set_shape((height, width, ch))
        return image

    def produce_stream_op(self, image_path, batch_size, config, image_type='train', reuse=None):
        image_files, image_annot_files = self.load_image(image_path, image_type)
        image_tensor = tf.convert_to_tensor(image_files)
        image_annot_tensor = tf.convert_to_tensor(image_annot_files)
        image_queue = tf.train.slice_input_producer([image_tensor, image_annot_tensor])
        pre_images = self.preprocess(tf.image.decode_image(tf.read_file(image_queue[0])), ch=3)
        pre_images_annot = self.preprocess(tf.image.decode_image(tf.read_file(image_queue[1])), ch=1, dtype=tf.uint8)

        images, images_annot = tf.train.batch([pre_images, pre_images_annot], batch_size=batch_size, allow_smaller_final_batch=True)

        image_annot_shape = images_annot.get_shape().as_list()
        images_annot = tf.reshape(images_annot, [batch_size, image_annot_shape[1], image_annot_shape[2]])
        images_onehot = tf.one_hot(images_annot, config['class_num'])

        logits, probabilities = self._back_bone.building_net(input=images,
                                                             batch_size=batch_size,
                                                             c=config['class_num'],
                                                             stage_two_three=config['stage_two_three'],
                                                             repeat_init_block=config['repeat_init_block'],
                                                             skip=config['skip'],
                                                             reuse=reuse)

        w = weight.weight.median_frequency_balancing(image_annot_files, config['class_num'])
        w = images_onehot * w
        w = tf.reduce_sum(w, axis=3)
        losses = tf.losses.softmax_cross_entropy(onehot_labels=images_onehot, logits=logits, weights=w)

        predict = tf.argmax(probabilities, axis=-1)
        accuracy, acc_update_op = tf.metrics.accuracy(images_annot, predict)
        mean_iou, iou_update_op = tf.metrics.mean_iou(images_annot, predict, num_classes=config['class_num'])
        metrics_op = tf.group(acc_update_op, iou_update_op)

        return losses, accuracy, mean_iou, metrics_op, int(len(image_files)/batch_size,)

    def train(self, config_path):
        network_config = config.get_config(config_path)

        train_losses, train_accuracy, train_mean_iou, train_metrics_op, train_step_num_per_epoch = self.produce_stream_op(network_config['image_path'], network_config['batch_size'], config=network_config)
        global_step = tf.train.get_or_create_global_step()
        decay_steps = network_config['num_epochs_before_decay'] * train_step_num_per_epoch
        learning_rate_dec = tf.train.exponential_decay(learning_rate=network_config['learning_rate'], global_step=global_step, decay_steps=decay_steps, decay_rate=network_config['decay_rate'], staircase=True)
        train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_dec, epsilon=network_config['epsilon'])
        train_op = slim.learning.create_train_op(train_losses, train_optimizer)

        val_losses, val_accuracy, val_mean_iou, val_metrics_op, val_step_num_per_epoch = self.produce_stream_op(network_config['image_path'], network_config['eval_batch_size'], image_type='val', config=network_config, reuse=True)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                for step in range(train_step_num_per_epoch * network_config['num_epoch']):
                    loss, global_step_cnt, acc, iou, update_op = sess.run([train_op, global_step, train_accuracy, train_mean_iou, train_metrics_op])
                    print('train epoch:{}-loss={},acc={},iou={}'.format(global_step_cnt, loss, acc, iou))

                    if step % min(network_config['batch_size'], train_step_num_per_epoch) == 0:
                        for i in range(val_step_num_per_epoch):
                            _, acc, iou = sess.run([val_metrics_op, val_accuracy, val_mean_iou])
                            print('val epoch:{}-acc={},iou={}'.format(step, acc, iou))
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
