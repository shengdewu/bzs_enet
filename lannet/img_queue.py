import os
import logging

class img_queue(object):
    def __init__(self, src_path):
        self._img_queue = self.__load_img(src_path)
        self._start_index = 0
        return

    def __load_img(self, src_path):
        src_img_files = list()
        with open(src_path, 'r') as handler:
            while True:
                line = handler.readline()
                if not line:
                    break
                path = line.strip('\n')

                if not os.path.exists(path):
                    logging.info('{} is not exists'.format(path))
                    raise FileExistsError('{} is not exists'.format(path))

                src_img_files.append(path)
        return src_img_files

    def is_empty(self):
        return len(self._img_queue) == 0

    def next_batch(self, batch, img_width, img_height):
        start = self._start_index
        self._start_index = self._start_index + batch
        return self._img_queue[start: self._start_index]