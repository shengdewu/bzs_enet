from enet.enet_block import enet_block
import tensorflow as tf
import tensorflow.contrib.slim as slim

class enet(object):
    def __init__(self):
        self._enet_block = enet_block()
        return

    def building_net(self, input, batch_size, c=12, stage_two_three=2, repeat_init_block=1, skip=False, reuse=None):

        inputs_shape = input.get_shape().as_list()
        input.set_shape(shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))

        with tf.variable_scope(name_or_scope='enet', reuse=reuse):
            with slim.arg_scope([self._enet_block.bottleneck, self._enet_block.bottleneck_upsample, self._enet_block.bottleneck_downsample], drop_prob=0.01),\
                 slim.arg_scope([slim.conv2d_transpose, slim.conv2d], activation_fn=None),\
                 slim.arg_scope([enet_block.prebn], fused=True):

                initial = self._enet_block.initial_block(input, scope='initial')
                for i in range(0, repeat_init_block):
                    initial = self._enet_block.initial_block(input, scope='initial'+str(i))

                unpool_indices = list()
                skip_net = list()
                #stage 1
                bottleneck, pool_indices = self._enet_block.bottleneck_downsample(initial, output_depth=64, filter_size=3, scope='bottleneck_1_0')
                unpool_indices.append((pool_indices, initial.get_shape().as_list()))
                skip_net.append(initial)

                for i in range(1, 5, 1):
                    bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=64, filter_size=3, scope='bottleneck_1.'+str(i))

                #stage 2.0
                up_pool = bottleneck
                bottleneck, pool_indices = self._enet_block.bottleneck_downsample(bottleneck, output_depth=128, filter_size=3, scope='bottleneck_2.0')
                unpool_indices.append((pool_indices, up_pool.get_shape().as_list()))
                skip_net.append(bottleneck)

            with slim.arg_scope([self._enet_block.bottleneck, self._enet_block.bottleneck_upsample, self._enet_block.bottleneck_downsample], drop_prob=0.1):
                #stage 2， 3
                for i in range(stage_two_three):
                    for stage in range(2, 4, 1):
                        bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, scope='bottleneck_{}.1_{}'.format(stage, i))
                        bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=2, scope='bottleneck_{}.2_{}'.format(stage, i))
                        bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=5, btype='decomposed', scope='bottleneck_{}.3_{}'.format(stage, i))
                        bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=4, scope='bottleneck_{}.4_{}'.format(stage, i))
                        bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, scope='bottleneck_{}.5_{}'.format(stage, i))
                        bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=8, scope='bottleneck_{}.6_{}'.format(stage, i))
                        bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=5, btype='decomposed', scope='bottleneck_{}.7_{}'.format(stage, i))
                        bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=128, filter_size=3, btype='dilation', dilation_rate=16, scope='bottleneck_{}.8_{}'.format(stage, i))

                #stage 4
                bottleneck = self._enet_block.bottleneck_upsample(bottleneck, output_shape=unpool_indices[1][1], pool_indices=unpool_indices[1][0], filter_size=3, scope='bottleneck_4.0')
                if skip:
                    bottleneck = tf.add(skip_net[1], bottleneck)

                bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=64, filter_size=3, scope='bottleneck_4.1')
                bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=64, filter_size=3, scope='bottleneck_4.2')

                #stage 5
                bottleneck = self._enet_block.bottleneck_upsample(bottleneck, output_shape=unpool_indices[0][1], pool_indices=unpool_indices[0][0], filter_size=3, scope='bottleneck_5.0')
                if skip:
                    bottleneck = tf.add(skip_net[0], bottleneck)
                bottleneck = self._enet_block.bottleneck(bottleneck, output_depth=16, filter_size=3, scope='bottleneck_5.1')

            #full conv
            logits = slim.conv2d_transpose(bottleneck, num_outputs=c, kernel_size=[2,2], stride=2, scope='fullconv')
            probabilities = slim.softmax(logits, scope='probabilities')

        return logits, probabilities
