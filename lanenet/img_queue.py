import os
import logging
import numpy as np
import cv2

class img_queue(object):
    def __init__(self, root_path, file_path):
        self._img_queue = self.__load_img(root_path, file_path)
        self._start_index = 0
        return

    def __load_img(self, root_path, file_path):
        src_img_files = list()
        with open(root_path + '/' + file_path, 'r') as handler:
            while True:
                line = handler.readline()
                if not line:
                    break
                path = root_path + '/' + line.strip('\n')

                if not os.path.exists(path):
                    logging.info('{} is not exists'.format(path))
                    raise FileExistsError('{} is not exists'.format(path))

                src_img_files.append(path)
        return src_img_files

    def is_empty(self):
        return len(self._img_queue) == 0

    def is_continue(self, batch):
        return len(self._img_queue) > self._start_index + batch

    def batch(self):
        return self._start_index

    def next_batch(self, batch, img_width, img_height):
        start = self._start_index
        self._start_index = self._start_index + batch
        files = self._img_queue[start: self._start_index]
        imgs = list()
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            img_resize = cv2.resize(img, (img_width, img_height))
            imgs.append(img_resize)
            del img
        return np.array(imgs)