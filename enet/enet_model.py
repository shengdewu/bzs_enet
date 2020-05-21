from enet.enet_block import enet_block
import tensorflow as tf
import tensorflow.contrib.slim as slim
from enet.nn import nn

class enet_model(object):
    def __init__(self):
        self._enet_block = enet_block()
        return

    def enet_one_stage(self, initial, unpool_indices, skip_net):
        with slim.arg_scope([self._enet_block.bottleneck, self._enet_block.bottleneck_upsample, self._enet_block.bottleneck_downsample], drop_prob=0.01):
            # stage 1
            bottleneck, pool_indices = self._enet_block.bottleneck_downsample(initial, output_depth=64, filter_size=3, scope='bottleneck_1_0')
            unpool_indices.append((pool_indices, initial.get_shape().as_list()))

            for i in range(1, 5, 1):
                bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=64, filter_size=3, scope='bottleneck_1.' + str(i))

            skip_net.append(bottleneck)
        return bottleneck

    def enet_two_0_stage(self, bottleneck, unpool_indices, skip_net):
        with slim.arg_scope([self._enet_block.bottleneck, self._enet_block.bottleneck_upsample, self._enet_block.bottleneck_downsample], drop_prob=0.1):
            # stage 2.0
            up_pool = bottleneck
            bottleneck, pool_indices = self._enet_block.bottleneck_downsample(bottleneck, output_depth=128, filter_size=3, scope='bottleneck_2.0')
            unpool_indices.append((pool_indices, up_pool.get_shape().as_list()))
        return bottleneck

    def enet_tow_three_stage(self, bottleneck, stage_two_three, stage):
        for i in range(stage_two_three):  # repeat section 2, without bottleneck2.0 that is refrence papers
            bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, scope='bottleneck_{}.1_{}'.format(stage, i))
            bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=2, scope='bottleneck_{}.2_{}'.format(stage, i))
            bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=5, btype='decomposed', scope='bottleneck_{}.3_{}'.format(stage, i))
            bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=4, scope='bottleneck_{}.4_{}'.format(stage, i))
            bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, scope='bottleneck_{}.5_{}'.format(stage, i))
            bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=8, scope='bottleneck_{}.6_{}'.format(stage, i))
            bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=5, btype='decomposed', scope='bottleneck_{}.7_{}'.format(stage, i))
            bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=16, scope='bottleneck_{}.8_{}'.format(stage, i))
        return bottleneck

    def enet_four_stage(self, bottleneck, unpool_indices, skip_net, skip):
        with slim.arg_scope([self._enet_block.bottleneck, self._enet_block.bottleneck_upsample, self._enet_block.bottleneck_downsample], drop_prob=0.1), slim.arg_scope([self._enet_block.prebn, nn.prelu], relu=True):
            # stage 4
            bottleneck = self._enet_block.bottleneck_upsample(bottleneck, output_shape=unpool_indices[1][1], pool_indices=unpool_indices[1][0], filter_size=3, scope='bottleneck_4.0')
            if skip:
                bottleneck = tf.add(skip_net[1], bottleneck, name='bottleneck_4_skip')

            bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=64, filter_size=3, scope='bottleneck_4.1')
            bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=64, filter_size=3, scope='bottleneck_4.2')
        return bottleneck

    def enet_five_stage(self, bottleneck, unpool_indices, skip_net, skip):
        with slim.arg_scope([self._enet_block.bottleneck, self._enet_block.bottleneck_upsample, self._enet_block.bottleneck_downsample], drop_prob=0.1), slim.arg_scope([self._enet_block.prebn, nn.prelu], relu=True):
            bottleneck = self._enet_block.bottleneck_upsample(bottleneck, output_shape=unpool_indices[0][1], pool_indices=unpool_indices[0][0], filter_size=3, scope='bottleneck_5.0')
            if skip:
                bottleneck = tf.add(skip_net[0], bottleneck, name='bottleneck_5_skip')
            bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=16, filter_size=3, scope='bottleneck_5.1')
        return bottleneck
