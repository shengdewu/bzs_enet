import os
import json
import cv2
import numpy as np
from log import log_configure
import logging
import math

class lanenet_data_pipline(object):
    def __init__(self):
        log_configure.log_configure.init_log('lanenet_data_porcess', os.getcwd()+'/process_data_log')
        return

    def _create_path(self, path):
        os.makedirs(path, exist_ok=True)
        return path

    def generate_data(self, data_path, out_path, rate=0.7):
        '''
        :param data_path: tuSimple 数据集路径
        :param out_path: 产生结果输出路径
        :param rate: 训练集和训练集的比例
        :return:
        '''

        if out_path[-1] == '/':
            out_path = out_path[0:-1]

        if not os.path.exists(data_path):
            raise FileExistsError('{} not find data path'.format(data_path))

        binary_path = self._create_path(out_path + '/gt_binary_img')
        instance_path = self._create_path(out_path + '/gt_instance_img')
        img_path = self._create_path(out_path + '/gt_src_img')

        json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]

        if len(json_files) < 0:
            raise FileExistsError('{} not exists json files'.format(data_path))

        instance_file_name = set()
        binary_file_name = set()
        src_file_name = set()

        total_files = list()

        for jfile in json_files:
            logging.info('start process json file: {}'.format(jfile))
            with open(data_path+'/'+jfile, 'r') as handle:
                while True:
                    line = handle.readline()
                    if not line:
                        break
                    line = line.strip('\n')
                    lane_dict = json.loads(line)
                    lanes = lane_dict['lanes']
                    h_samples = lane_dict['h_samples']
                    raw_file = lane_dict['raw_file']
                    file_seg = raw_file.split('/')
                    file_name = file_seg[-3] + '-' + file_seg[-2]+'-' + file_seg[-1]
                    image_path = '{}/{}'.format(data_path, raw_file)
                    logging.info('start process image {}'.format(image_path))

                    if not os.path.exists(image_path):
                        logging.warning('{} not exists'.format(image_path))
                        continue

                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    binary_img = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
                    instance_img = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)

                    for index in range(len(lanes)):
                        lane = lanes[index]
                        if len(lane) != len(h_samples):
                            raise Exception('tusimple lane data error len(lane) != len(h_samples)/({}!={})'.format(len(lane), len(h_samples)))

                        x = list()
                        y = list()
                        for i in range(len(lane)):
                            if lane[i] != -2:
                                x.append(lane[i])
                                y.append(h_samples[i])

                        line = np.array([x, y]).transpose()

                        cv2.polylines(binary_img, [line], color=255, thickness=5, isClosed=False, lineType=cv2.FILLED)
                        cv2.polylines(instance_img, [line], color=50 + 50 * index, thickness=5, isClosed=False, lineType=cv2.FILLED)

                    binary_save_path = '{}/{}'.format(binary_path, file_name)
                    instance_save_path = '{}/{}'.format(instance_path, file_name)
                    img_save_path = '{}/{}'.format(img_path, file_name)

                    if binary_save_path in binary_file_name:
                        logging.warning('{} is exists'.format(binary_save_path))
                        raise FileExistsError('{} is exists'.format(binary_save_path))

                    if instance_save_path in instance_file_name:
                        logging.warning('{} is exists'.format(instance_save_path))
                        raise FileExistsError('{} is exists'.format(instance_save_path))

                    if img_save_path in src_file_name:
                        logging.warning('{} is exists'.format(img_save_path))
                        raise FileExistsError('{} is exists'.format(img_save_path))

                    cv2.imwrite(binary_save_path, binary_img)
                    cv2.imwrite(instance_save_path, instance_img)
                    cv2.imwrite(img_save_path, image)

                    # cv2.imshow('binary', binary_img)
                    # cv2.imshow('src', image)
                    # cv2.imshow('instance', instance_img)
                    # cv2.waitKey(10)
                    binary_file_name.add(binary_save_path)
                    instance_file_name.add(instance_save_path)
                    src_file_name.add(img_save_path)

                    total_files.append((binary_save_path, instance_save_path, img_save_path))

        train_len = math.ceil(len(total_files) * 0.7)
        with open(out_path+'/train_files.txt', 'w') as thandle:
            for index in range(train_len):
                thandle.write(total_files[index][0]+' '+total_files[index][1]+' '+total_files[index][2]+'\n')

        with open(out_path+'/test_files.txt', 'w') as thandle:
            for index in range(train_len+1, len(total_files)):
                thandle.write(total_files[index][0]+' '+total_files[index][1]+' '+total_files[index][2]+'\n')

        return

if __name__ == '__main__':
    lanenet_data_provide = lanenet_data_pipline()
    lanenet_data_provide.generate_data('D:/work_space/tuSimpleDataSet/train/', 'D:/work_space/tuSimpleDataSet/lanenet_data/')