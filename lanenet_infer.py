from lanenet.lanenet_predict import lanenet_predict
from config.config import config
from log.log_configure import log_configure
import sys

if __name__ == '__main__':
    lannet_model = lanenet_predict()
    network_config = config.get_config(sys.argv[1])
    log_configure.init_log('lanenet_infer', network_config['log_path'])

    lannet_model.infer(network_config)
    print('infer success!!')