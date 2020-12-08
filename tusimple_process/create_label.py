import numpy as np
import tusimple_process.ultranet_comm

class tusimple_label:
    def __init__(self):
        self._row_anchors = tusimple_process.ultranet_comm.ROW_ANCHORS
        self._cells = tusimple_process.ultranet_comm.CELLS
        self._lanes = tusimple_process.ultranet_comm.LANES
        return

    def find_start_pos(self, row_sampel, start_line):
        l, r = 0, len(row_sampel)-1
        pos = 0
        while True:
            mid = int((l + r) / 2)

            if row_sampel[mid] > start_line:
                r = mid
            if row_sampel[mid] < start_line:
                l = mid
            if row_sampel[mid] == start_line:
                pos = mid
                break
            if r - l == 1:
                pos = r
                break
        return pos

    def get_index(self, label):
        h, w = label.shape
        row_anchors = self._row_anchors
        if h != 720:
            row_anchors = [int(i*h/720) for i in self._row_anchors]

        all_idx = np.zeros((self._lanes, len(row_anchors), 2)) # x,y
        for i, r in enumerate(row_anchors):
            label_r = label[int(round(r))]
            for lane_idx in range(1, 5):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = np.mean(pos)

        #将到图像底部[n , 720]的车道线补出
        all_idx_cp = all_idx.copy()
        for i in range(self._lanes):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue
            valid = all_idx_cp[i, :, 1] != -1
            valid_idx = all_idx_cp[i, valid, :]
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                continue
            if len(valid_idx) < 6:
                continue
            valid_idx_half = valid_idx[len(valid_idx)//2:, :]
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
            pos = self.find_start_pos(all_idx_cp[i, :, 0], valid_idx_half[-1, 0]) + 1
            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > w-1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted

        return all_idx_cp

    def grid_pts(self, lane_pts, gride_num, w):
        num_lanes, row_anchors, c = lane_pts.shape
        col_sample = np.linspace(0, w-1, gride_num)
        assert c == 2
        to_pts = np.zeros((row_anchors, num_lanes))
        for i in range(num_lanes):
            pti = lane_pts[i, :, 1]
            to_pts[:, i] = np.asarray([int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else gride_num for pt in pti])
        return to_pts.astype(int)

    def get_segment(self, label):
        np.resize(label, new_shape=(36, 100)).astype(np.int32)
        return

    def create_label(self, label_img, src_img):
        # 待定，是否先进行变换
        lane_pts = self.get_index(label_img)
        h, w, c = src_img.shape
        cls_label = self.grid_pts(lane_pts, self._cells, w)
        return src_img, cls_label