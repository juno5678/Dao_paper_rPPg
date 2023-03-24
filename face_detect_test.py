from face_detection import FaceDetection
from video_rgb_only import Video_RGB
from image_rgb import Image_RGB
import cv2
import sys

if __name__ == '__main__':

    input_rgb = Image_RGB()
    dataPath = sys.argv[1]
    input_rgb.dirname = dataPath
    print(sys.argv[1])
    fd = FaceDetection()
    input_rgb.start()
    t = 0
    while 1:

        rgb_frame = input_rgb.get_frame()
        if t == 0:
            rgb_face = fd.face_detect(rgb_frame)
        else:
            rgb_face = fd.face_track(rgb_frame)
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        rgb_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2BGR)
        cv2.imshow('rgb', rgb_frame)
        cv2.imshow('rgb face', rgb_face)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
        t += 1
