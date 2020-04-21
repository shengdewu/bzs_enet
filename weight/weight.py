import numpy as np
from PIL import Image
import tensorflow as tf

def median_frequency_balancing(labeles_path, class_num):
    '''
    note: we weight each pixel by αc = median freq/freq(c)
          where freq(c) is (the number of pixels of class c) divided by (the total number of pixels in images where c is present),
          and (median freq is the median of these frequencies)

        "the number of pixels of class c": Represents the total number of pixels of class c across all images of the dataset.
        "The total number of pixels in images where c is present": Represents the total number of pixels across all images (where there is at least one pixel of class c) of the dataset.
        "median frequency is the median of these frequencies": Sort the frequencies calculated above and pick the median.

    :param labeles_path:
    :param class_num:
    :return:
    '''
    freq_class = dict()
    for c in range(class_num):
        freq_class[c] = list()

    for path in labeles_path:
        image = np.array(Image.open(path))
        for c in range(class_num):
            mask = np.equal(image, c).astype(np.float32)
            freq = np.sum(mask)
            if freq > 0:
                freq_class[c].append(freq)

    total = 0
    for c, freq in freq_class.items():
        total += np.sum(freq)

    mbf = list()
    for c, freq in freq_class.items():
        median_freq = np.median(freq)/np.sum(freq)
        freq_c = np.sum(freq) / total
        median_blance_freq = median_freq / freq_c
        mbf.append(median_blance_freq)

    mbf[-1] = 0.0
    return mbf


def inverse_class_probability_weighting( label, num):
    '''
    note: the inverse class probability weighting cite from enet a deep neural network architecture for real-time semantic segmentation
        define as :
            w = 1/ln(c + p)  s.t c=1.02
    :param self:
    :param label: 类别类
    :param num:  类别总数
    :return:
    '''

    freq = dict()
    for n in range(num):
        freq[n] = tf.reduce_sum(tf.equal(label, n))

    total = tf.size(label)

    w = list()
    for c, cnt in freq.items():
        w.append(tf.divide(1.0, tf.log(1.02+cnt/total)))
    return w
