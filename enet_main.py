import sys
from config.config import config
from log.log_configure import log_configure
from lannet.enet_train import enet_train

if __name__ == '__main__':
    lannet_model = enet_train()
    network_config = config.get_config(sys.argv[1])
    log_configure.init_log('enet_train', network_config['log_path'])

    lannet_model.train(network_config)
    print('train success!!')
