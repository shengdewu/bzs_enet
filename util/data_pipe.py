import tensorflow as tf
import multiprocessing

class data_pipe():
    def __init__(self, num_parallel_calls=None):
        self._num_parallel_calls = num_parallel_calls if num_parallel_calls is not None else multiprocessing.cpu_count()
        print('data pipe map thread {}'.format(self._num_parallel_calls))
        return

    def make_pipe(self, batch_size, tensor_tuple, pretreatment=None):
        '''
        :param batch_size:
        :param tensor_tuple: 队列元组
        :param pretreatment: 后处理
        :return:
        '''
        data_set = tf.data.Dataset.from_tensor_slices(tensor_tuple).repeat(count=None)
        if pretreatment is not None:
            data_set = data_set.map(pretreatment, num_parallel_calls=self._num_parallel_calls)
        data_set = data_set.batch(batch_size, drop_remainder=True).prefetch(batch_size * self._num_parallel_calls)
        iter = data_set.make_one_shot_iterator().get_next()
        return iter
