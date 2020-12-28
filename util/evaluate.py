import tensorflow as tf

class evalute(object):
    def __init__(self):
        return

    @staticmethod
    def accuracy(label, predict):
        logit = tf.argmax(tf.nn.softmax(predict, axis=-1), axis=-1)
        logit = tf.expand_dims(tf.cast(logit, tf.float16), axis=-1)

        true_idx = tf.where(tf.equal(logit, 1))
        acc = tf.count_nonzero(tf.gather_nd(label, true_idx))

        total = tf.shape(tf.gather_nd(label, tf.where(tf.equal(label, 1))))[0]

        return tf.divide(tf.cast(acc, tf.float16), tf.cast(total, tf.float16))

    @staticmethod
    def fn(label, predict):
        logit = tf.argmax(tf.nn.softmax(predict, axis=-1), axis=-1)
        logit = tf.expand_dims(tf.cast(logit, tf.float16), axis=-1)

        true_idx = tf.where(tf.equal(label, 1))
        miss = tf.shape(tf.gather_nd(logit, true_idx))[0]
        miss = miss - tf.cast(tf.count_nonzero(tf.gather_nd(logit, true_idx)), tf.int32)

        total = tf.shape(tf.gather_nd(label, tf.where(tf.equal(label, 1))))[0]

        return tf.divide(tf.cast(miss, tf.float16), tf.cast(total, tf.float16))
