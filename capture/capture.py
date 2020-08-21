import cv2

class capture():

    def __init__(self, place=0, display=False):
        '''
        :param place: 0 内置； 1 外置
        '''
        self._display = display
        # self._cap = cv2.VideoCapture(place)
        # if not self._cap.isOpened():
        #     raise RuntimeError("can't open camera (0 itself; 1 usb) {}".format(place))
        return

    # def capture(self):
    #     ret, frame = self._cap.read()
    #     if ret:
    #         if self._display:
    #             cv2.imshow('capture', frame)
    #             cv2.waitKey(1)
    #         return frame
    #     return None

    def test(self):
        frame = cv2.imread('D:/work_space/tuSimpleDataSetSource/test\clips/0530/1492626047222176976_0/1.jpg', cv2.IMREAD_COLOR)
        if self._display:
            cv2.imshow('capture', frame)
            cv2.waitKey(1)
        return frame



