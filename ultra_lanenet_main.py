import sys
from config.config import config
from log.log_configure import log_configure
from ultra_lane.ultra_lane import ultra_lane

if __name__ == '__main__':
    ultra_model = ultra_lane()
    network_config = config.get_config(sys.argv[1])
    log_configure.init_log('ultra_train', network_config['log_path'])

    ultra_model.train(network_config)
    print('train success!!')
