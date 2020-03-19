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
        if not relu:
            alpha = tf.get_variable(scope + 'alpha', x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            negative = alpha * (x - abs(x)) * 0.5
            positive = tf.nn.relu(x)
            active = positive + negative
        else:
            active = tf.nn.relu(x, name=scope)
        return active

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


    @staticmethod
    def enet_arg_scope(weight_decay=2e-4,
                       batch_norm_decay=0.1,
                       batch_norm_epsilon=0.001):
        '''
        The arg scope for enet model. The weight decay is 2e-4 as seen in the paper.
        Batch_norm decay is 0.1 (momentum 0.1) according to official implementation.

        INPUTS:
        - weight_decay(float): the weight decay for weights variables in conv2d and separable conv2d
        - batch_norm_decay(float): decay for the moving average of batch_norm momentums.
        - batch_norm_epsilon(float): small float added to variance to avoid dividing by zero.

        OUTPUTS:
        - scope(arg_scope): a tf-slim arg_scope with the parameters needed for xception.
        '''
        # Set weight_decay for weights in conv2d and separable_conv2d layers.
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_regularizer=slim.l2_regularizer(weight_decay)):
            # Set parameters for batch_norm.
            with slim.arg_scope([slim.batch_norm],
                                decay=batch_norm_decay,
                                epsilon=batch_norm_epsilon) as scope:
                return scope



