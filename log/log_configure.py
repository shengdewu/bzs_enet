import logging
import os

class log_configure(object):

    @staticmethod
    def init_log(log_name, log_path):
        if not os.path.isdir(log_path):
            os.makedirs(log_path)

        logging.basicConfig(
            filename=log_path + '/' + log_name + '.log',
            format='<%(levelname)s %(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s> %(message)s',
            level=logging.DEBUG)
        return

