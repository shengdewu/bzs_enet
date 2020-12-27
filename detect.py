from lanenet.lanenet_realtime import lanenet_realtime
from capture.capture import capture
from config.config import config
from log.log_configure import log_configure
import sys
import cv2

if __name__ == '__main__':
    capture_pool = capture(place=1)

    lannet_model = lanenet_realtime(capture_pool.capture)
    network_config = config.get_config(sys.argv[1])
    log_configure.init_log('detect', network_config['log_path'])

    lannet_model.start(network_config)
    print('detect success!!')