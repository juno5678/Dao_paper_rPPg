from face_detection import FaceDetection
from face_segmentation import FaceSegmentation
from video_realsense_file import Video
import sys
import cv2

if __name__ == '__main__':

    dataPath = sys.argv[1]

    input_realsense = Video()
    input_realsense.dirname = dataPath
    fd = FaceDetection()
    fs = FaceSegmentation()
    input_realsense.start()
    count = 0
    while 1:
        color_frame, nir_frame = input_realsense.get_frame()

        if count == 0:
            color_face = fd.face_detect(color_frame)
            #cv2.imshow("2",color_face)
            #cv2.waitKey(0)
            color_face, RGB_black_point = fs.face_segment(color_face)
            if nir_frame is not None:
                nir_face = fd.face_detect(nir_frame)
                nir_face, nir_black_point = fs.face_segment(nir_face)

        else:
            #color_face = self.fd.face_detect_rgb(rgb_frame)
            color_face = fd.face_track_rgb(color_frame)
            color_face, RGB_black_point = fs.face_segment(color_face)
            if nir_frame is not None:
                #nir_face = self.fd.face_detect_gray(nir_frame)
                nir_face = fd.face_track_gray(nir_frame)
                nir_face, nir_black_point = fs.face_segment(nir_face)

        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
        color_face = cv2.cvtColor(color_face, cv2.COLOR_RGB2BGR)
        cv2.imshow("color frame", color_frame)
        cv2.imshow("color face", color_face)
        cv2.imshow("nir frame", nir_frame)
        cv2.imshow("nir face", nir_face)
        if cv2.waitKey(3) == 27:
            cv2.destroyAllWindows()
            break






