from enet import enet
from enet.enet_block import enet_block
import tensorflow.contrib.slim as slim
import tensorflow as tf

class lanenet_model(object):
    def __init__(self):
        self._enet_block = enet_block()
        return

    def front_backbone(self, input, batch_size,  scope, reuse=None, is_trainging=True):
        input_shape = input.get_shape().as_list()
        input.set_shape(shape=[batch_size, input_shape[1], input_shape[2], input_shape[2]])

        with tf.variable_scope(name_or_scope=scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], activation_fn=None), slim.arg_scope(enet_block.prebn, fused=True, is_trainging=is_trainging):
                skip_net = list()
                unpool_indices = list()
                initial = self._enet_block.initial_block(input, scope='initial')

                skip_net.append(initial)

                with slim.arg_scope([self._enet_block.bottleneck, self._enet_block.bottleneck_upsample, self._enet_block.bottleneck_downsample], drop_prob=0.01):
                    #stage 1
                    bottleneck, pool_indices = self._enet_block.bottleneck_downsample(initial, output_depth=64, filter_size=3, scope='bottleneck_1_0')
                    unpool_indices.append((pool_indices, initial.get_shape().as_list()))

                    for i in range(1, 5, 1):
                        bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=64, filter_size=3, scope='bottleneck_1.'+str(i))

                    skip_net.append(bottleneck)

                with slim.arg_scope([self._enet_block.bottleneck, self._enet_block.bottleneck_upsample, self._enet_block.bottleneck_downsample], drop_prob=0.1):
                    #stage 2.0
                    up_pool = bottleneck
                    bottleneck, pool_indices = self._enet_block.bottleneck_downsample(bottleneck, output_depth=128, filter_size=3, scope='bottleneck_2.0')
                    unpool_indices.append((pool_indices, up_pool.get_shape().as_list()))

                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, scope='bottleneck_{}.1_{}'.format(2, 1))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=2, scope='bottleneck_{}.2_{}'.format(2, 2))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=5, btype='decomposed', scope='bottleneck_{}.3_{}'.format(2, 3))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=4, scope='bottleneck_{}.{}'.format(2, 4))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, scope='bottleneck_{}.5_{}'.format(2, 5))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=8, scope='bottleneck_{}.}'.format(2, 6))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=5, btype='decomposed', scope='bottleneck_{}.{}'.format(2, 7))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=16, scope='bottleneck_{}.{}'.format(2, 8))
        return bottleneck, skip_net, unpool_indices

    def back_backbone(self, input, skip_net, unpool_indices, c, scope, skip=False, reuse=None, is_trainging=True):
        with tf.variable_scope(name_or_scope=scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], activation_fn=None), slim.arg_scope(enet_block.prebn, fused=True, is_trainging=is_trainging):
                with slim.arg_scope([self._enet_block.bottleneck, self._enet_block.bottleneck_upsample, self._enet_block.bottleneck_downsample], drop_prob=0.1):
                    #stage 3.0
                    bottleneck = self._enet_block.bottleneck(input, output_depth=128, filter_size=3, scope='bottleneck_{}.1_{}'.format(3, 1))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=2, scope='bottleneck_{}.2_{}'.format(3, 2))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=5, btype='decomposed', scope='bottleneck_{}.3_{}'.format(3, 3))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=4, scope='bottleneck_{}.{}'.format(3, 4))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, scope='bottleneck_{}.5_{}'.format(3, 5))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=8, scope='bottleneck_{}.}'.format(3, 6))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=5, btype='decomposed', scope='bottleneck_{}.{}'.format(3, 7))
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=16, scope='bottleneck_{}.{}'.format(3, 8))

                with slim.arg_scope([self._enet_block.bottleneck, self._enet_block.bottleneck_upsample,
                                     self._enet_block.bottleneck_downsample], drop_prob=0.1), slim.arg_scope(
                        [self._enet_block.prebn], relu=True):
                    # stage 4
                    bottleneck = self._enet_block.bottleneck_upsample(bottleneck, output_shape=unpool_indices[1][1],
                                                                      pool_indices=unpool_indices[1][0],
                                                                      filter_size=3, scope='bottleneck_4.0')
                    if skip:
                        bottleneck = tf.add(skip_net[1], bottleneck, name='bottleneck_4_skip')

                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=64, filter_size=3,
                                                             scope='bottleneck_4.1')
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=64, filter_size=3,
                                                             scope='bottleneck_4.2')

                    # stage 5
                    bottleneck = self._enet_block.bottleneck_upsample(bottleneck, output_shape=unpool_indices[0][1],
                                                                      pool_indices=unpool_indices[0][0],
                                                                      filter_size=3, scope='bottleneck_5.0')
                    if skip:
                        bottleneck = tf.add(skip_net[0], bottleneck, name='bottleneck_5_skip')
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=16, filter_size=3,
                                                             scope='bottleneck_5.1')

            # full conv
            logits = slim.conv2d_transpose(bottleneck, num_outputs=c, kernel_size=[2, 2], stride=2,
                                           scope='fullconv')
        return logits

    def build_net(self, input,  batch_size, skip=False, reuse=None, is_trainging=True):

        front, skip_net, unpool_indices = self.front_backbone(input, batch_size, 'front_backbone', reuse, is_trainging)

        binary_logits = self.back_backbone(front, skip_net, unpool_indices, 2, "binary", skip, reuse, is_trainging)

        embedding_logits = self.back_backbone(front, skip_net, unpool_indices, 4, "embedding", skip, reuse, is_trainging)

        return binary_logits, embedding_logits


