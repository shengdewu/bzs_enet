from enet.enet_block import enet_block
import tensorflow.contrib.slim as slim
import tensorflow as tf
from enet import nn
from enet import enet_model

class lanenet_model(object):
    def __init__(self):
        self._enet_model = enet_model.enet_model()
        return

    def front_backbone(self, input, batch_size,  scope, reuse=None, is_training=True):
        input_shape = input.get_shape().as_list()
        input.set_shape(shape=[batch_size, input_shape[1], input_shape[2], input_shape[3]])

        skip_net = list()
        unpool_indices = list()

        with tf.variable_scope(name_or_scope=scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], activation_fn=None), slim.arg_scope([enet_block.prebn], fused=True, is_training=is_training):
                initial = self._enet_model.enent_init_stage(input, skip_net)
                bottleneck = self._enet_model.enet_one_stage(initial, unpool_indices, skip_net)
                bottleneck = self._enet_model.enet_two_0_stage(bottleneck, unpool_indices, skip_net)
                bottleneck = self._enet_model.enet_tow_three_stage(bottleneck, 1, 2)

        return bottleneck, skip_net, unpool_indices

    def back_backbone(self, input, skip_net, unpool_indices, c, scope, skip=False, reuse=None, is_training=True):
        with tf.variable_scope(name_or_scope=scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], activation_fn=None), slim.arg_scope([enet_block.prebn], fused=True, is_training=is_training):
                bottleneck = self._enet_model.enet_tow_three_stage(input, 1, 3)
                bottleneck = self._enet_model.enet_four_stage(bottleneck, unpool_indices, skip_net, skip)
                bottleneck = self._enet_model.enet_five_stage(bottleneck, unpool_indices, skip_net, skip)

            # full conv
            logits = slim.conv2d_transpose(bottleneck, num_outputs=c, kernel_size=[2, 2], stride=2, scope='fullconv')
        return logits

    def build_net(self, input,  batch_size, l2_weight_decay, skip=False, reuse=None, is_trainging=True):
        with slim.arg_scope(nn.nn.enet_arg_scope(weight_decay=l2_weight_decay)):
            front, skip_net, unpool_indices = self.front_backbone(input, batch_size, 'front_backbone', reuse, is_trainging)

            binary_logits = self.back_backbone(front, skip_net, unpool_indices, 2, "binary", skip, reuse, is_trainging)

            embedding_logits = self.back_backbone(front, skip_net, unpool_indices, 4, "embedding", skip, reuse, is_trainging)

            with tf.variable_scope(name_or_scope='pix_embedding', reuse=reuse):
                pix_embedding = slim.conv2d(embedding_logits, 4, [1, 1], stride=1, scope='conv2d')
                pix_embedding = tf.identity(pix_embedding)
                pix_embedding = nn.nn.prelu(pix_embedding, 'relu', relu=True)

        return binary_logits, pix_embedding


