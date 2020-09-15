import cv2

class video_capture():
    def __init__(self, video_stream):
        self._video_stream = video_stream
        self._close = False
        return

    def close(self):
        self._close = True

    def pool(self):
        cap = cv2.VideoCapture(0)
        while not self._close:
            sucess, img = cap.read()
            self._video_stream(img)
            cv2.imshow("img", img)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
        cap.release()
        return

if __name__ == '__main__':
    vcap = video_capture()
    vcap.pool()