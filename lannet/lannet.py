from enet.enet import enet
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

class lannet(object):
    def __init__(self):
        self._back_bone = enet()
        return


    def load_image(self, image_path, image_type='train'):
        image_files = sorted([str('{}/{}/{}').format(image_path, image_type, file) for file in os.listdir(str('{}/{}').format(image_path, image_type))])
        annot_image_files = sorted([str('{}/{}annot/{}').format(image_path, image_type, file)  for file in os.listdir(str('{}/{}annot').format(image_path, image_type))])
        return image_files, annot_image_files

    def preprocess(self, image, height=360, width=480, ch=3, dtype=tf.float32):
        if image.dtype != dtype:
            image = tf.image.convert_image_dtype(image, dtype=dtype)
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)
        image.set_shape((height, width, ch))
        return image

    def produce_stream_op(self, image_path, batch_size, image_type='train', class_num=12, reuse=None):
        image_files, image_annot_files = self.load_image(image_path, image_type)
        image_tensor = tf.convert_to_tensor(image_files)
        image_annot_tensor = tf.convert_to_tensor(image_annot_files)
        image_queue = tf.train.slice_input_producer([image_tensor, image_annot_tensor])
        pre_images = self.preprocess(tf.image.decode_png(tf.read_file(image_queue[0])), ch=3)
        pre_images_annot = self.preprocess(tf.image.decode_png(tf.read_file(image_queue[1])), ch=1, dtype=tf.uint8)

        images, images_annot = tf.train.batch([pre_images, pre_images_annot], batch_size=batch_size, allow_smaller_final_batch=False)

        image_annot_shape = images_annot.get_shape().as_list()
        images_annot = tf.reshape(images_annot, [batch_size, image_annot_shape[1], image_annot_shape[2]])
        images_onehot = tf.one_hot(images_annot, class_num)

        logits, probabilities = self._back_bone.building_net(input=images, batch_size=batch_size, reuse=reuse)
        predict = tf.argmax(probabilities, axis=-1)

        losses = tf.losses.softmax_cross_entropy(images_onehot, logits)
        accuracy, acc_update_op = tf.metrics.accuracy(images_annot, predict)
        mean_iou, iou_update_op = tf.metrics.mean_iou(images_annot, predict, num_classes=class_num)
        metrics_op = tf.group(acc_update_op, iou_update_op)

        return losses, accuracy, mean_iou, metrics_op, len(image_files)

    def train(self):
        batch_size = 10
        class_num = 12
        learning_rate = 0.0001
        decay_steps = 100
        decay_rate = 0.99
        epsilon = 0.0001
        image_path = 'E:/workspace/TensorFlow-ENet/dataset'

        train_losses, train_accuracy, train_mean_iou, train_metrics_op, train_total_images = self.produce_stream_op(image_path, batch_size, class_num=class_num)
        val_losses, val_accuracy, valmean_iou, val_metrics_op, val_total_images = self.produce_stream_op(image_path, batch_size, image_type='val', class_num=class_num, reuse=True)

        global_step = tf.train.get_or_create_global_step()
        learning_rate_dec = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
        train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_dec, epsilon=epsilon)
        train_op = slim.learning.create_train_op(train_losses, train_optimizer)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                for epoch in range(int(train_total_images/batch_size)):
                    _, loss, acc, iou, update_op = sess.run([train_op, train_losses, train_accuracy, train_mean_iou, train_metrics_op])
                    print('train epoch:{}-loss={},acc={},iou={}'.format(epoch, loss, acc, iou))

                    acc = sess.run(val_accuracy)
                    print('val epoch:{}-acc={}'.format(epoch, acc))
            except Exception as err:
                print('{}'.format(err))
            finally:
                coord.request_stop()
                coord.join(threads)
            coord.join(threads)


        print('test')
        return
