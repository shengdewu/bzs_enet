import tensorflow as tf
import tensorflow.contrib.slim as slim

class common_block():
    @staticmethod
    def add(identify, x, scope, reuse=True):
        with tf.variable_scope(scope, reuse=reuse):
            _, xw, xh, xc = list(x.get_shape())
            _, iw, ih, ic = list(identify.get_shape())

            identify_map = identify
            if xc != ic:
                if xw != iw:
                    identify_map = slim.conv2d(identify, ic, [3, 3], 2, padding='SAME', reuse=reuse, scope=scope+'add-dimensions-adjust')
                identify_map = slim.conv2d(identify_map, xc, [1, 1], stride=1, padding='SAME', reuse=reuse, scope=scope+'add-dimensions')

            o = tf.add(identify_map, x)
        return o

    @staticmethod
    def relu(x, scope):
        o = tf.nn.relu(x, name=scope+'-relue')
        return o

    @staticmethod
    @slim.add_arg_scope
    def conv(x, stride, ochannel, scope, reuse=True, ksize=3, trainable=True):
        with tf.variable_scope(scope, reuse=reuse):
            _, iw, ih, ic = list(x.get_shape())
            o = slim.conv2d(x, ochannel, [ksize, ksize], stride, padding='SAME', scope='conv2d')
            o = slim.batch_norm(o, is_training=trainable, fused=True, scope='bn')
        return o

    @staticmethod
    @slim.add_arg_scope
    def conv1x1(x, ochannel, scope, reuse=True, trainable=True):
        with tf.variable_scope(scope, reuse=reuse):
            o = slim.conv2d(x, ochannel, [1, 1], 1, padding='SAME', scope='conv2d')
            o = slim.batch_norm(o, is_training=trainable, fused=True, scope='convbn')
        return o

    @staticmethod
    def fc(x, scope, num_class, reuse=True):
        with tf.variable_scope(scope, reuse=reuse):
            o = tf.nn.avg_pool2d(x, 7, strides=1, padding='VALID', name='avgpool')

            o = slim.conv2d(o, num_class, [1, 1], 1, padding='SAME', reuse=reuse, scope='1000d-fc')

            o = tf.nn.softmax(o, name='softmax')
        return o

class base_block():
    def __init__(self):
        self.stride = 0
        self.ksize = 3
        return

    def block(self, x, scope, stride, ochannel, reuse=True, trainable=True):
        with tf.variable_scope(scope, reuse=reuse):
            o = common_block.conv(x, stride, ochannel=ochannel, ksize=3, scope='cov1', reuse=reuse,trainable=trainable)
            o = common_block.relu(o, 'relu')
            o = common_block.conv(o, 1, ochannel=ochannel, ksize=3, scope='cov2', reuse=reuse, trainable=trainable)
        return o

    def make_layer(self, x, scope, block_num, downsample, ochannel, reuse=True, trainable=True):

        stride = 1
        if downsample:
            stride = 2

        o = self.block(x, scope+'-block1', stride, ochannel, reuse, trainable)
        o = common_block.relu(o, scope + '-block1-conv2-relu')
        for i in range(1, block_num):
            o = self.block(o, scope + '-block2', 1, ochannel, reuse, trainable)
            if i < block_num-1:
                o = common_block.relu(o, scope + '-block' + str(i + 1) + '-relu')

        o = common_block.add(x, o, scope+'-add', reuse)
        o = common_block.relu(o, scope + '-add-relu')
        return o

class bottleneck_block():
    def __init__(self):
        self.stride = 0
        self.ksize = 3
        return

    def block(self, x, scope, stride, dim_reduce_channel, dim_ascend_channel, reuse=True, trainable=True):
        with tf.variable_scope(scope, reuse=reuse):
            o = common_block.conv1x1(x, ochannel=dim_reduce_channel, scope='cov1x1-reduce', reuse=reuse, trainable=trainable)
            o = common_block.relu(o, 'relu')
            o = common_block.conv(o, stride, ochannel=dim_reduce_channel, ksize=3, scope='cov1', reuse=reuse, trainable=trainable)
            o = common_block.relu(o, 'relu')
            o = common_block.conv1x1(o, ochannel=dim_ascend_channel, scope='cov1x1-ascend', reuse=reuse, trainable=trainable)
        return o

    def make_layer(self, x, scope, downsample, block_num, dim_reduce_channel, dim_ascend_channel, reuse=True, trainable=True):
        stride = 1
        if downsample:
            stride = 2

        o = self.block(x, scope+'-block1', stride, dim_reduce_channel, dim_ascend_channel, reuse, trainable)
        o = common_block.relu(o, scope+'-block1-relu')
        for i in range(1, block_num):
            o = self.block(o, scope + '-block'+str(i+1), 1, dim_reduce_channel, dim_ascend_channel, reuse, trainable)
            if i < block_num-1:
                o = common_block.relu(o, scope + '-block' + str(i + 1) + '-relu')

        o = common_block.add(x, o, scope+'-add', reuse)
        return o


class resnet():
    def __init__(self):
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.layer5 = None
        return

    def __resnet_head(self, x, scope, trainable=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            o = slim.conv2d(x, 64, [7, 7], 2, padding='SAME', reuse=reuse, scope='head-conv2d')
            o = slim.batch_norm(o, is_training=trainable, fused=True, scope='head-bn')
            o = tf.nn.relu(o, name='head-relu')
            o = slim.max_pool2d(o, kernel_size=[3, 3], stride=2, padding='SAME', scope='head-maxpool')
        return o

    def resnet18(self, x, num_class, trainable=True, reuse=False):
        block = base_block()
        self.layer1 = self.__resnet_head(x, 'layer1', trainable, reuse)
        self.layer2 = block.make_layer(self.layer1, 'layer2', 2, False, 64, reuse, trainable)
        self.layer3 = block.make_layer(self.layer2, 'layer3', 2, True, 128, reuse, trainable)
        self.layer4 = block.make_layer(self.layer3, 'layer4', 2, True, 256, reuse, trainable)
        self.layer5 = block.make_layer(self.layer4, 'layer5', 2, True, 512, reuse, trainable)
        return common_block.fc(self.layer5, 'layer-fc', num_class, reuse)

    def resnet50(self, x, num_class, trainable=True, reuse=False):
        block = bottleneck_block()
        self.layer1 = self.__resnet_head(x, 'layer1', trainable, reuse)
        self.layer2 = block.make_layer(self.layer1, 'layer2', False, 3, 64, 256, reuse, trainable)
        self.layer3 = block.make_layer(self.layer2, 'layer3', True, 4, 128, 512, reuse, trainable)
        self.layer4 = block.make_layer(self.layer3, 'layer4', True, 6, 256, 1024, reuse, trainable)
        self.layer5 = block.make_layer(self.layer4, 'layer5', True, 3, 512, 2048, reuse, trainable)
        return common_block.fc(self.layer5, 'layer-fc', num_class, reuse)

if __name__ == '__main__':
    resnet_model = resnet()
    x = tf.get_variable(name='x',shape=(10, 224, 224, 3))
    #model18 = resnet_model.resnet18(x, 1000)
    model50 = resnet_model.resnet50(x, 1000)
    print('resnet50')


