import tensorflow as tf
import os
import logging

class data_queue():
    def __init__(self, width, height):
        self._height = height
        self._width = width
        return

    def _featch_img_paths(self, file_path, root_path):
        binary_img_files = list()
        instance_img_files = list()
        src_img_files = list()
        with open(file_path, 'r') as handler:
            while True:
                line = handler.readline()
                if not line:
                    break
                path = line.strip('\n')
                pathes = [root_path+'/'+ p for p in path.split(' ')]
                for p in pathes:
                    if not os.path.exists(p):
                        logging.info('{} is not exists'.format(path))
                        raise FileExistsError('{} is not exists'.format(path))

                src_img_files.append(pathes[0])
                binary_img_files.append(pathes[1])
                instance_img_files.append(pathes[2])

        return binary_img_files, instance_img_files, src_img_files

    def make_queue(self, root_path, batch_size, sub_path='train_files.txt'):
        file_path = root_path + '/' + sub_path
        binary_img_files, instance_img_files, src_img_files = self._featch_img_paths(file_path, root_path)

        binary_img_tensor = tf.convert_to_tensor(binary_img_files)
        instance_img_tensor = tf.convert_to_tensor(instance_img_files)
        src_img_tensor = tf.convert_to_tensor(src_img_files)

        img_producer = tf.train.slice_input_producer([src_img_tensor, binary_img_tensor, instance_img_tensor])

        src_img = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(img_producer[0]), channels=3), self._height, self._width)
        binary_img = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(img_producer[1]), channels=1), self._height, self._width)
        instance_img = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(img_producer[2]), channels=1), self._height, self._width)

        binary_img = tf.divide(binary_img, tf.reduce_max(binary_img))
        binary_img = tf.cast(binary_img, tf.uint8)

        src_img = tf.cast(src_img, tf.float32)

        return tf.train.batch([src_img, binary_img, instance_img], batch_size=batch_size, allow_smaller_final_batch=True)
