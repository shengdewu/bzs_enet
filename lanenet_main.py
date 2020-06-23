import sys
from config.config import config
from log.log_configure import log_configure
from lanenet.lanenet_train import lanenet_train

if __name__ == '__main__':
    lannet_model = lanenet_train()
    network_config = config.get_config(sys.argv[1])
    log_configure.init_log('lanenet_train', network_config['log_path'])

    lannet_model.train(network_config)
    print('train success!!')
