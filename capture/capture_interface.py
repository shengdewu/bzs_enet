import cv2
import threading

class capture_interface(threading.Thread):

    def __init__(self, target, args=(), place=0, max_retry=10, display=False):
        '''
        :param target:
        :param args:
        :param place: 0 内置； 1 外置
        '''
        threading.Thread.__init__(self, name='video-capture')

        self._max_retry = max_retry
        self._target = target
        self._args = args
        self._display = display
        self._cap = cv2.VideoCapture(place)
        if not self._cap.isOpened():
            raise RuntimeError("can't open camera (0 itself; 1 usb) {}".format(place))
        return

    def start_capture(self):
        self.start()
        self.join()
        return

    def stop_capture(self):
        self._cap.release()
        return

    def run(self):
        max_retry = 0
        while True:
            ret, frame = self._cap.read()
            if ret:
                if self._display:
                    cv2.imshow('capture', frame)
                    cv2.waitKey(1)
                self._target(frame, *self._args)
            else:
                max_retry += 1
                if max_retry > self._max_retry:
                    self.stop_capture()
                    break
        return


