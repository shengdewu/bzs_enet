from enet.enet_block import enet_block
import tensorflow as tf
import tensorflow.contrib.slim as slim
from enet.enet_model import enet_model

class enet(object):
    def __init__(self):
        self._enet_model = enet_model()
        return

    def building_net(self, input, batch_size, c=12, stage_two_three=1, skip=False, reuse=None, is_trainging=True):

        inputs_shape = input.get_shape().as_list()
        input.set_shape(shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))

        with tf.variable_scope(name_or_scope='enet', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], activation_fn=None), slim.arg_scope([enet_block.prebn], fused=True), slim.arg_scope([enet_block.prebn], is_training=is_trainging):

                skip_net = list()
                unpool_indices = list()

                initial = self._enet_model.enent_init_stage(input)
                skip_net.append(initial)

                bottleneck = self._enet_model.enet_one_stage(initial, unpool_indices, skip_net)
                bottleneck = self._enet_model.enet_two_0_stage(bottleneck, unpool_indices, skip_net)
                for i in range(stage_two_three):
                    for stage in range(2, 4, 1):
                        bottleneck = self._enet_model.enet_tow_three_stage(bottleneck, stage, i)
                bottleneck = self._enet_model.enet_four_stage(bottleneck, unpool_indices, skip_net, skip)
                bottleneck = self._enet_model.enet_five_stage(bottleneck, unpool_indices, skip_net, skip)

            #full conv
            logits = slim.conv2d_transpose(bottleneck, num_outputs=c, kernel_size=[2,2], stride=2, scope='fullconv')
            probabilities = slim.softmax(logits, scope='probabilities')

        return logits, probabilities
