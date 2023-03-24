from webcam import Webcam
from webcam import Camera_RGB
import cv2
if __name__ == '__main__':
    input = Webcam()
    input.start()
    while 1:
        color_frame , nir_frame = input.get_frame()
        cv2.imshow('color', color_frame)
        cv2.imshow('nir', nir_frame)
        if cv2.waitKey(1) == 27:
            break;

    #cam = cv2.VideoCapture(1)
    #while True:
    #    ret, img = cam.read()
    #    vis = img.copy()
    #    cv2.imshow('color img',vis)
    #    if 0xFF & cv2.waitKey(5) == 27:
    #        break
    #cv2.destroyAllWindows()


