import tensorflow as tf
import os
import logging
import multiprocessing

class test_data_pipe():
    def __init__(self, width, height, num_parallel_calls=None):
        self._height = height
        self._width = width
        self._num_parallel_calls = num_parallel_calls if num_parallel_calls is not None else multiprocessing.cpu_count()
        print('width {}, height {}, map thread {}'.format(self._width, self._height, self._num_parallel_calls))
        return

    def _featch_img_path(self, file_path, root_path):
        src_img_files = list()
        with open(file_path, 'r') as handler:
            while True:
                line = handler.readline()
                if not line:
                    break

                path = root_path+'/'+ line.strip('\n')
                if not os.path.exists(path):
                    logging.info('{} is not exists'.format(path))
                    raise FileExistsError('{} is not exists'.format(path))

                src_img_files.append(path)

        return src_img_files

    def _pre_process_img(self, src_img_tensor):
        src_img = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(src_img_tensor), channels=3), self._height, self._width)

        src_img = tf.cast(src_img, tf.float32)

        return src_img

    def make_pipe(self, root_path, batch_size, sub_path):
        file_path = root_path + '/' + sub_path
        src_img_files = self._featch_img_path(file_path, root_path)
        src_img_tensor = tf.convert_to_tensor(src_img_files)

        data_set = tf.data.Dataset.from_tensor_slices(src_img_tensor)
        data_set = data_set.map(self._pre_process_img, num_parallel_calls=self._num_parallel_calls)
        data_set = data_set.batch(batch_size, drop_remainder=True).prefetch(batch_size)
        iter = data_set.make_one_shot_iterator().get_next()
        return iter


