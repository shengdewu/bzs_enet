import tensorflow as tf
import tensorflow.contrib.slim as slim
from enet.nn import nn

class enet_block(object):
    def __init__(self):
        self._sub_branch = 'sub_branch'
        self._main_branch = 'main_branch'
        return

    @slim.add_arg_scope
    def prebn(self, input, scope, is_training=True, fused=None, relu=False):
        net_conv = slim.batch_norm(input, is_training=is_training, fused=fused, scope=scope + '_batch_norm')
        output = nn.prelu(net_conv, relu=relu, scope=scope+'_prelu')
        return output

    @slim.add_arg_scope
    def initial_block(self, input, scope):
        conv = slim.conv2d(input, 13, [3,3], stride=2, scope=scope+'_conv2d')
        bn = self.prebn(conv, scope=scope)

        max_pool = slim.max_pool2d(input, kernel_size=[2,2], stride=2, scope=scope+'_max_pool2d')

        output = tf.concat([bn, max_pool], axis=3, name=scope+'_concat')
        return output

    @slim.add_arg_scope
    def bottleneck_downsample(self, input, output_depth, filter_size, scope, projection=4, drop_prob=0.01, is_training=True):
        input_shape = input.get_shape().as_list()
        project_depth = int(input_shape[3] / projection)

        #first 1*1
        sub_net = slim.conv2d(input, project_depth, [2, 2], stride=2, scope=scope+'_conv2d_1')
        sub_net = self.prebn(sub_net, is_training=is_training, scope=scope+'_prebn1')

        #conv
        sub_net = slim.conv2d(sub_net, project_depth, [filter_size, filter_size], stride=1, scope=scope+'_conv2d_2')
        sub_net = self.prebn(sub_net, is_training=is_training, scope=scope+'_prebn_2')

        #second 1*1
        sub_net = slim.conv2d(sub_net, output_depth, [1, 1], stride=1, scope=scope+'_conv2d_3')
        sub_net = self.prebn(sub_net, is_training=is_training, scope=scope+'_prebn_3')

        #regularizer
        sub_net = nn.spatial_dropout(sub_net, drop_prob=drop_prob, seed=1, scope=scope+'_regular')
        # sub_net = nn.prelu(sub_net, scope=scope+'prelu1')

        #main branch
        main_net, indices = tf.nn.max_pool_with_argmax(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope+'_main_branch_max_pooling')
        pad_depth = abs(output_depth - main_net.get_shape().as_list()[3])
        padding = tf.convert_to_tensor([[0,0], [0,0], [0,0], [0,pad_depth]])
        main_net = tf.pad(main_net, paddings=padding, name=scope+'_main_branch_pad')

        net = tf.add(main_net, sub_net)
        output = nn.prelu(net, scope=scope+'_prelu')

        return output, indices

    @slim.add_arg_scope
    def bottleneck_upsample(self, input, output_shape, filter_size, scope, pool_indices, projection=4, drop_prob=0.01, is_training=True):

        input_shape = input.get_shape().as_list()
        project_depth = int(input_shape[3] / projection)
        #first 1*1
        sub_net = slim.conv2d(input, project_depth, [1, 1], stride=1, scope=scope+'_conv2d_1')
        sub_net = self.prebn(sub_net, is_training=is_training, scope=scope+'_prebn1')

        #conv
        filter_shape = [filter_size, filter_size, output_shape[-1], project_depth]
        filter_shape_tensor = tf.get_variable(shape=filter_shape, initializer=tf.constant_initializer, dtype=tf.float32, name=scope+'_transpose_shape')
        sub_net = tf.nn.conv2d_transpose(sub_net, filter=filter_shape_tensor, output_shape=output_shape, strides=[1, 2, 2, 1], name=scope+'_transpose')
        sub_net = self.prebn(sub_net, is_training=is_training, scope=scope+'_prebn2')

        #second 1*1
        sub_net = slim.conv2d(sub_net, output_shape[-1], [1, 1], stride=1, scope=scope+'_conv2d_2')
        sub_net = self.prebn(sub_net, is_training=is_training, scope=scope+'_prebn3')

        #regularizer
        sub_net = nn.spatial_dropout(sub_net, drop_prob=drop_prob, seed=1, scope=scope+'_regular')
        sub_net = nn.prelu(sub_net, scope=scope+'_prelu1')

        #main branch
        main_net = slim.conv2d(input, output_shape[-1], [1, 1], stride=1, scope=scope + '_main_conv2d')
        main_net = self.prebn(main_net, scope=scope + '_main_branch')
        main_net = nn.unpool(main_net, pool_indices, output_shape=output_shape, scope=scope + '_main_branch_unpool')

        net = tf.add(main_net, sub_net)
        output = nn.prelu(net, scope=scope+'_prelu2')

        return output

    @slim.add_arg_scope
    def bottleneck(self, input, output_depth, filter_size, scope, btype='normal', dilation_rate=1, projection=4, drop_prob=0.01, is_training=True):
        opt = ['decomposed', 'dilation', 'normal']

        if btype not in opt:
            raise RuntimeError('not found vaid bottlenekc in ["decomposed", "dilation", "normal"]')

        project_depth = int(input.get_shape().as_list()[3] / projection)

        #first 1*1
        sub_net = slim.conv2d(input, project_depth, [1, 1], stride=1, scope=scope+'_conv2d_1')
        sub_net = self.prebn(sub_net, is_training=is_training, scope=scope+'_prebn1')

        #conv
        if btype == 'decomposed':
            decomposed_1 = [1, filter_size]
            decomposed_2 = [filter_size, 1]
            sub_net = slim.conv2d(sub_net, project_depth, decomposed_1, rate=dilation_rate, stride=1, scope=scope + '_conv2d_2')
            sub_net = slim.conv2d(sub_net, project_depth, decomposed_2, rate=dilation_rate, stride=1, scope=scope + '_conv2d_3')
        else:
            sub_net = slim.conv2d(sub_net, project_depth, [filter_size, filter_size], rate=dilation_rate, stride=1, scope=scope+'_conv2d_2')

        sub_net = self.prebn(sub_net, is_training=is_training, scope=scope+'_prebn2')

        #second 1*1
        sub_net = slim.conv2d(sub_net, output_depth, [1, 1], stride=1, scope=scope+'_conv2d_4')
        sub_net = self.prebn(sub_net, is_training=is_training, scope=scope+'_prebn3')

        #regularizer
        sub_net = nn.spatial_dropout(sub_net, drop_prob=drop_prob, seed=1, scope=scope+'_regular')
        sub_net = nn.prelu(sub_net, scope=scope+'_prelu1')

        #main branch
        net = tf.add(input, sub_net)
        output = nn.prelu(net, scope=scope+'_prelu2')
        return output
