import tensorflow as tf
import tensorflow.contrib.slim as slim

class hnet():
    def __init__(self):
        return

    def hnet_model(self, tensor_in, name, out_channel,is_training=True):
        conv1 = slim.conv2d(tensor_in, out_channel, 3, 1, 'SAME', scope=name+'-conv1')
        bn1 = slim.batch_norm(conv1, is_training=is_training, scope=name+'-bn1')
        relu1 = tf.nn.relu(bn1, name=name+'-relu1')

        conv2 = slim.conv2d(relu1, out_channel, 3, 1, 'SAME', scope=name+'-conv2')
        bn2 = slim.batch_norm(conv2, is_training=is_training, scope=name+'-bn2')
        relu2 = tf.nn.relu(bn2, name=name+'-relu2')

        mp1 = slim.max_pool2d(relu2, kernel_size=[2, 2], stride=2, scope=name+'-max-pool2d1')
        return mp1

    def create_hnet(self, input_tensor, is_training=True):
        hnet1 = self.hnet_model(input_tensor, name='hnet-1', out_channel=16, is_training=is_training)
        hnet2 = self.hnet_model(hnet1, name='hnet-2',out_channel=32, is_training=is_training)
        hnet3 = self.hnet_model(hnet2, name='hnet-3',out_channel=64, is_training=is_training)
        liner = tf.layers.dense(hnet3, 1024, name='liner-1')
        bn = slim.batch_norm(liner, is_training=is_training, scope='liner-1-bn')
        relu = tf.nn.relu(bn, name='liner-relu2')
        return tf.layers.dense(relu, 6)

    def cost(self, h, input_image):
        indices = tf.constant([[0,0], [0,1], [0,2], [1,4], [1,5], [2,7], [2,8]])
        shape = tf.constant([3,3])
        h_mat = tf.scatter_nd(indices, tf.concat([h, [1]]), shape)

        lan_mask = tf.where(input_image, 1)
        pro_img = tf.matmul(h_mat, input_image)
        return
