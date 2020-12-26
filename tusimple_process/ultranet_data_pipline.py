import os
import json
from shutil import copyfile
import numpy as np
import cv2
import tqdm
from tusimple_process.create_label import tusimple_label
import random
import math

class ultranet_data_pipline:
    def __init__(self, cls_label=False):
        self.cls_label_handle = None
        if cls_label: #是否同时构建标签
            self.cls_label_handle = tusimple_label()
        return

    def _create_path(self, path):
        os.makedirs(path, exist_ok=True)
        return path

    def calc_k(self, line, min_len):
        x = line[0::2]
        y = line[1::2]
        length = np.sqrt((x[0]-x[-1])**2 + (y[0] - y[-1])**2)
        if length < min_len:
            return -20
        p = np.polyfit(x, y, deg=1)
        return np.arctan(p[0])

    def draw_lane(self, im, line, idx, show=False):
        '''
        Generate the segmentation label according to json annotation
        '''
        line_x = line[::2]
        line_y = line[1::2]
        pt0 = (int(line_x[0]), int(line_y[0]))
        if show:
            cv2.putText(im, str(idx), (int(line_x[len(line_x) // 2]), int(line_y[len(line_x) // 2]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            idx = idx * 60

        for i in range(len(line_x) - 1):
            cv2.line(im, pt0, (int(line_x[i + 1]), int(line_y[i + 1])), (idx,), thickness=16)
            pt0 = (int(line_x[i + 1]), int(line_y[i + 1]))

    def draw(self, lines, shape, min_len, show=False):
        ks = np.array([self.calc_k(line, min_len) for line in lines])
        ks = ks[ks != -20]
        k_neg = ks[ks < 0].copy()
        k_pos = ks[ks > 0].copy()
        k_neg.sort()
        k_pos.sort()

        label_image = np.zeros(shape, dtype=np.uint8)  # 越陡峭越靠中间
        bin_label = [0, 0, 0, 0]
        if len(k_neg) == 1:
            which = np.where(ks == k_neg[0])[0][0]  # 1
            self.draw_lane(label_image, lines[which], 2, show)
            bin_label[1] = 1
        elif len(k_neg) == 2:
            which = np.where(ks == k_neg[1])[0][0]  # 1
            self.draw_lane(label_image, lines[which], 1, show)
            bin_label[0] = 1
            which = np.where(ks == k_neg[0])[0][0]  # 2
            self.draw_lane(label_image, lines[which], 2, show)
            bin_label[1] = 1
        elif len(k_neg) > 2:  # 只取靠近中间的
            which = np.where(ks == k_neg[1])[0][0]  # 1
            self.draw_lane(label_image, lines[which], 1, show)
            bin_label[0] = 1
            which = np.where(ks == k_neg[0])[0][0]  # 2
            self.draw_lane(label_image, lines[which], 2, show)
            bin_label[1] = 1

        if len(k_pos) == 1:
            which = np.where(ks == k_pos[0])[0][0]
            self.draw_lane(label_image, lines[which], 3, show)
            bin_label[2] = 1
        elif len(k_pos) == 2:
            which = np.where(ks == k_pos[1])[0][0]  # 3
            self.draw_lane(label_image, lines[which], 3, show)
            bin_label[2] = 1
            which = np.where(ks == k_pos[0])[0][0]  # 4
            self.draw_lane(label_image, lines[which], 4, show)
            bin_label[3] = 1
        elif len(k_pos) > 2:
            which = np.where(ks == k_pos[-1])[0][0]  # 3
            self.draw_lane(label_image, lines[which], 3, show)
            bin_label[2] = 1
            which = np.where(ks == k_pos[-2])[0][0]  # 4
            self.draw_lane(label_image, lines[which], 4, show)
            bin_label[3] = 1

        # if show:
        #     cv2.imshow("label", label_image)
        #     cv2.waitKey(3)
        #     cv2.destroyAllWindows()
        return label_image, bin_label

    def generate_data(self, data_path, out_path, show=False, shape=(720, 1280), rate=0.7):
        '''
        :param data_path: tuSimple 数据集路径
        :param out_path: 产生结果输出路径
        :param shape: 图像实际的 h,w
        :param min_len: 车道的最小长度
        :return:
        '''

        if out_path[-1] != '/':
            out_path = out_path + '/'

        if not os.path.exists(data_path):
            raise FileExistsError('{} not find data path'.format(data_path))

        out_img_path = out_path + '/' + 'img'
        self._create_path(out_img_path)

        json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]

        if len(json_files) < 0:
            raise FileExistsError('{} not exists json files'.format(data_path))

        total_files = list()

        for jfile in tqdm.tqdm(json_files):
            with open(data_path + '/' + jfile, 'r') as handle:
                while True:
                    line = handle.readline()
                    if not line:
                        break
                    line = line.strip('\n')
                    lane_dict = json.loads(line)
                    lanes = lane_dict['lanes']
                    h_sample = np.array(lane_dict['h_samples'])
                    raw_file = lane_dict['raw_file']
                    image_path = '{}/{}'.format(data_path, raw_file)
                    file_seg = raw_file.split('/')
                    label_name = file_seg[-3] + '-' + file_seg[-2]+'-' + file_seg[-1][0:file_seg[-1].rfind('.')]+'.png'
                    image_name = file_seg[-3] + '-' + file_seg[-2]+'-' + file_seg[-1]

                    if not os.path.exists(image_path):
                        print('{} not exists'.format(image_path))
                        continue

                    lines = list()
                    for index in range(len(lanes)):
                        if len(lanes[index]) != h_sample.shape[0]:
                            raise Exception('tusimple lane data error len(lane) != len(h_samples)/({}!={})'.format(len(lanes[index]), h_sample.shape[0]))

                        lane = np.array(lanes[index])
                        if np.all(lane == -2):
                            print('current lane {}/{} is invalid'.format(jfile, image_path))
                            continue
                        valid = lane != -2
                        line_tmp = [None] * (h_sample[valid].shape[0] + lane[valid].shape[0])
                        line_tmp[0::2] = lane[valid]
                        line_tmp[1::2] = h_sample[valid]
                        lines.append(line_tmp)

                    label_image, bin_label = self.draw(lines, shape, 90, show)
                    if self.cls_label_handle is None:
                        label_out_path = out_img_path + '/' + label_name
                        cv2.imwrite(label_out_path, label_image)
                        image_out_path = out_img_path + '/' + image_name
                        copyfile(image_path, image_out_path)
                        total_files.append('img'+'/'+image_name + ' ' + 'img'+'/'+label_name + ' ' + ''.join(list(map(str, bin_label))) + '\n')
                    else:
                        src_img = cv2.imread(image_path)
                        h, w, c = src_img.shape
                        cls_label = self.cls_label_handle.create_label(label_image, w)
                        self.cls_label_handle.rescontruct(cls_label, src_img, True)
                        src_img = cv2.resize(src_img, dsize=(800, 288), interpolation=cv2.INTER_LINEAR)
                        label_image = cv2.resize(label_image, dsize=(800, 288), interpolation=cv2.INTER_NEAREST)
                        self.cls_label_handle.rescontruct(cls_label, src_img, True)

                        cls_name = label_name[0:label_name.rfind('.')] + '-cls.png'
                        cv2.imwrite(out_img_path + '/' + label_name, label_image)
                        cv2.imwrite(out_img_path + '/' + cls_name, cls_label)
                        cv2.imwrite(out_img_path + '/' + image_name, src_img)
                        total_files.append('img/'+image_name + ' ' + 'img/'+label_name + ' ' + 'img/' + cls_name + ' ' + ''.join(list(map(str, bin_label))) + '\n')

        random.shuffle(total_files)
        train_len = math.ceil(len(total_files) * rate)
        with open(out_path+'/train_files.txt', 'w') as train_handle:
            for index in range(train_len):
                train_handle.write(total_files[index])

        with open(out_path+'/valid_files.txt', 'w') as test_handle:
            for index in range(train_len+1, len(total_files)):
                test_handle.write(total_files[index])

        return

if __name__ == '__main__':
    lanenet_data_provide = ultranet_data_pipline(True)
    lanenet_data_provide.generate_data('D:/work_space/tuSimpleDataSetSource/train/', 'D:/work_space/tuSimpleDataSet/train_utlra/train/')