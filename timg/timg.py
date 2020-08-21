import cv2
import numpy as np

class timg():
    def __init__(self):
        return

    @staticmethod
    def crop_pad(img, target_height, target_width):
        '''
        :param img: [h, w, c]
        :param target_height:
        :param target_width:
        :return:
        '''

        shape = img.shape
        w = shape[1]
        h = shape[0]

        #crop
        crop_width = (w - target_width) // 2
        crop_height = (h - target_height) // 2

        new_img = img.copy()
        if crop_width > 0:
            new_img = img[:,crop_width:w-crop_width]

        if crop_height > 0:
            new_img = new_img[crop_height:h-crop_height]

        # pad
        pad_width = -crop_width
        pad_height = -crop_height

        if pad_width > 0:
            new_img = np.pad(new_img, ((0,0), (pad_width, pad_width)), 'constant', constant_values=(0,))

        if pad_height > 0:
            new_img = np.pad(new_img, ((pad_height, pad_height), (0, 0)), 'constant', constant_values=(0,))

        return new_img