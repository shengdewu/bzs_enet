from lannet.enet_segment import enet_segment
import sys
from config.config import config
from log.log_configure import log_configure

if __name__ == '__main__':
    lannet_model = enet_segment()
    network_config = config.get_config(sys.argv[1])
    log_configure.init_log('enet', network_config['log_path'])

    lannet_model.train(network_config)
    print('train success!!')