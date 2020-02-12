import tensorflow as tf
import tensorflow.contrib.slim as slim

class nn(object):
    def __init__(self):
        return

    @staticmethod
    @slim.add_arg_scope
    def prelu(x, scope, relu=False):
        '''
        :param x:
        :param relu: bool
        :return:
        '''
        positive = tf.nn.relu(x, name=scope)
        negative = 0

        if not relu:
            alpha = tf.get_variable(scope + 'alpha', x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            negative = alpha * (x - abs(x)) * 0.5

        return positive + negative

    @staticmethod
    @slim.add_arg_scope
    def spatial_dropout(x, drop_prob, seed, scope):
        input_shape = x.get_shape().as_list()
        keep_prob = 1-drop_prob
        noise_shape = tf.constant(value=[input_shape[0], 1, 1, input_shape[-1]])
        drop = tf.nn.dropout(x, keep_prob, noise_shape, seed=seed, name=scope)
        return drop

    @staticmethod
    @slim.add_arg_scope
    def unpool(updates, mask, output_shape=None, scope='', k_size=[1, 2, 2, 1]):
        '''
        Unpooling function based on the implementation by Panaetius at https://github.com/tensorflow/tensorflow/issues/2169

        INPUTS:
        - inputs(Tensor): a 4D tensor of shape [batch_size, height, width, num_channels] that represents the input block to be upsampled
        - mask(Tensor): a 4D tensor that represents the argmax values/pooling indices of the previously max-pooled layer
        - k_size(list): a list of values representing the dimensions of the unpooling filter.
        - output_shape(list): a list of values to indicate what the final output shape should be after unpooling
        - scope(str): the string name to name your scope

        OUTPUTS:
        - ret(Tensor): the returned 4D tensor that has the shape of output_shape.

        '''
        with tf.variable_scope(scope):
            mask = tf.cast(mask, tf.int32)
            input_shape = tf.shape(updates, out_type=tf.int32)
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0], input_shape[1] * k_size[1], input_shape[2] * k_size[2], input_shape[3])

            # calculation indices for batch, height, width and feature maps
            one_like_mask = tf.ones_like(mask, dtype=tf.int32)
            batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
            batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[
                2]  # mask % (output_shape[2] * output_shape[3]) // output_shape[3]
            feature_range = tf.range(output_shape[3], dtype=tf.int32)
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
            values = tf.reshape(updates, [updates_size])

            ret = tf.scatter_nd(indices, values, output_shape)
            return ret





