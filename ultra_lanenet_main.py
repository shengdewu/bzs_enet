import sys
from config.config import config
from log.log_configure import log_configure
from ultra_lane.ultra_lane import ultra_lane
import tensorflow as tf

if __name__ == '__main__':
    ultra_model = ultra_lane()
    # x = tf.get_variable(name='x', shape=(10, 800, 288, 3))
    # net = ultra_model.make_net(x, 100, 56, 4)

    network_config = config.get_config(sys.argv[1])
    log_configure.init_log('ultra_train', network_config['result_path']+'/log')

    ultra_model.train(network_config)
    print('train success!!')
