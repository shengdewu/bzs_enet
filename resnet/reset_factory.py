import tensorflow as tf
import tensorflow.contrib.slim as slim
from resnet.resnet_block import resnet_block

class factory():
    def __init__(self):
        self._layer ={'18':[(64, 2), (128, 2), (256, 2), (512, 2)],
                      '34':[(64, 3), (128, 4), (256, 6), (512, 3)],
                      '50':[(64, 256, 3), (128,512, 4), (256, 1024, 6), (512, 2048, 3)],
                      '101':[(64, 256, 3), (128,512, 4), (256, 1024, 23), (512, 2048, 3)],
                      '152':[(64, 256, 3), (128,512, 8), (256, 1024, 36), (512, 2048, 3)]}

        self._block ={'18':'bottleneck_block',
                      '34':'bottleneck_block',
                      '50':'bottleneck_block_11',
                      '101':'bottleneck_block_11',
                      '152':'bottleneck_block_11'}
        return

    def create(self, x, layer, reuse=None, trainable=True):
        layer_param = self._layer.get(layer, None)
        block_type = self._block.get(layer, None)
        if layer_param is None or block_type is None:
            raise RuntimeError('invalid param {}'.format(layer))

        o = slim.conv2d(x, 64, [7, 7], 2, padding='SAME', reuse=reuse, scope='reset{}-conv1-conv'.format(layer))
        o = slim.batch_norm(o, is_training=trainable, fused=True, scope='reset{}-conv1-bn'.format(layer))
        o = tf.nn.relu(o, name='reset{}-conv1-relu'.format(layer))
        o = slim.max_pool2d(o, kernel_size=[3, 3], stride=2, padding='SAME', scope='reset{}-conv1-maxpool'.format(layer))

        if block_type == 'bottleneck_block_11':
            index = 2
            for lp in layer_param:
                oc = dict()
                oc['channel1'] = lp[0]
                oc['channel2'] = lp[1]
                oc['ksize'] = 3
                oc['stride'] = 1
                for i in range(lp[2]):
                    o = resnet_block.bottleneck_block_11(o, oc, name='reset{}-conv{}-'.format(layer,str(index)+'_'+str(i)))
                index += 1
        elif block_type == 'bottleneck_block':
            index = 2
            for lp in layer_param:
                oc = dict()
                oc['channel'] = lp[0]
                oc['ksize'] = 3
                oc['stride'] = 1
                for i in range(lp[1]):
                    o = resnet_block.bottleneck_block_11(o, oc, name='reset{}-conv{}-'.format(layer, str(index) + '_' + str(i)))
                index += 1
        else:
            raise RuntimeError('invalid block type')

        o = tf.nn.avg_pool2d(o, 7, stride=1, padding='VALID', name='reset{}-avgpool'.format(layer))

        o = slim.conv2d(o, 1000, [1, 1], 1, padding='SAME', reuse=reuse, scope='reset{}-fc'.format(layer))

        return tf.nn.softmax(o, name='reset{}-softmax'.format(layer))