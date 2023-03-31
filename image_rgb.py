import cv2
import numpy as np
import os
import sys


class Image_RGB(object):
    def __init__(self):
        self.dirname = ""
        self.img_list = []
        self.sequence_n = 0
        self.sequence_length = 0
        self.valid = False

    def start(self):
        print("Start image sequence" + self.dirname)
        if os.path.isdir(self.dirname):
            #print("is folder")
            self.sequence_n = 0
            self.img_list = [img for img in os.listdir(self.dirname) if img.endswith(".png")]
            self.img_list.sort()
            self.frame_length = len(self.img_list)
            self.valid = True
            if self.img_list:
                print("has image sequence")
        else:
            self.valid = False
            print("invalid folder!")
            return

    def stop(self):
        print("Video Stopped")

    def get_frame(self):
        if self.valid:
            if self.sequence_n < self.frame_length:
                frame = cv2.imread(os.path.join(self.dirname, self.img_list[self.sequence_n]))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.sequence_n += 1
            else:
                print("End of video")
                self.stop()
                return None
        else:
            frame = np.ones((480, 640, 3), dtype=np.uint8)
            col = (0, 256, 256)
            cv2.putText(frame, "(Error: Can not load the video)",
                        (65, 220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame
