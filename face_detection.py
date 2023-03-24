import numpy as np

import dlib
import cv2
from imutils import face_utils
# import numpy as np
# import face_alignment


class FaceDetection(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        #self.tracker1 = cv2.TrackerCSRT_create()
        #self.tracker2 = cv2.TrackerCSRT_create()
        self.tracker1 = cv2.TrackerMIL_create()
        self.tracker2 = cv2.TrackerMIL_create()
        self.rgb_roi = [0, 0, 0, 0]
        self.rgb_bbox = [0, 0, 0, 0]
        self.rgb_diff_bbox = [0, 0, 0, 0]
        self.rgb_diff_roi_bbox = [0, 0, 0, 0]
        self.rgb_pre_bbox = [0, 0, 0, 0]
        self.rgb_not_found_count = 0
        self.rgb_first_detect = True
        self.nir_roi = [0, 0, 0, 0]
        self.nir_bbox = [0, 0, 0, 0]
        self.nir_diff_bbox = [0, 0, 0, 0]
        self.nir_diff_roi_bbox = [0, 0, 0, 0]
        self.nir_pre_bbox = [0, 0, 0, 0]
        self.nir_not_found_count = 0
        self.nir_first_detect = True
        self.w_weights = 2
        self.h_weights = 2
         

    def face_detect(self, frame):
        if frame is None:
            print("No frame to do face detection")
            return
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.copy()
        rects = self.detector(gray, 1)
        if len(rects) > 0:
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            # juno adjust
            face_frame = frame[max(y-int(0.5*h),0):min(y+int(h*1),frame.shape[0]), x:min(x+int(w*1), frame.shape[1])].copy()
            bbox = (max(x-int(w*0.2), 0), max(y-int(h*0.2), 0), int(w*1.4), int(h*1.4))
            # shin adjust
            #face_frame = frame[max(y-int(0.225*h), 0):min(y+int(h*1.075), frame.shape[0]),
            #                   max(x-int(w*0.025), 0):min(x+int(w*1.025), frame.shape[1])].copy()
            #bbox = (max(x-int(w*0.075), 0), max(y-int(h*0.255), 0), int(w*1.05), int(h*1.3))
            #face_frame = frame[max(y-int(0.4*h), 0):min(y+int(h * 1), frame.shape[0]),
            #                   max(x-int(0.2*w), 0):min(x+int(w*1.2), frame.shape[1])].copy()
            #bbox = (max(x-int(w*0.2), 0), max(y-int(h*0.4), 0), int(w*1.4), int(h*1.4))
            #print("detect success")
            # bbox = (x, y, w, h)
            #self.tracker2.init(frame, bbox)
            if len(frame.shape) == 3:
                self.tracker1.init(frame, bbox)
            else:
                self.tracker2.init(frame, bbox)    
        else:
            print("failed detect face")
            return None

        return face_frame, rects

    def face_track(self, frame):
        if frame is None:
            print("No frame to do face tracking")
            return

        ok, bbox = self.tracker1.update(frame)
        if ok:
            frame = frame[bbox[1]:bbox[1]+int(bbox[3]), bbox[0]:bbox[0]+bbox[2]]
            #print("track : ", bbox[0], bbox[1], bbox[2], bbox[3])
        else:
            print("Update Tracker failure")
            return
        return frame

    def face_detect_rgb(self, frame):

        if frame is None:
            print("No frame to do face detection")
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.rgb_first_detect:
            rects = self.detector(gray, 1)
            if len(rects) > 0:
                (x, y, w, h) = face_utils.rect_to_bb(rects[0])
                #self.roi = (max(0, x - round(0.52*w)), max(0, y-round(0.3*h+0.5*1.3*h)), round(w*2), round(h*1.3*2))
                self.rgb_roi = (max(0, x-round(w * ((self.w_weights-1)/2))), max(0, y-round(h*((self.h_weights-1)/2))),
                                round(w*self.w_weights), round(h*self.h_weights))
                self.rgb_first_detect = False
                #print('roi : ', self.rgb_roi)
        if not self.rgb_first_detect:
            gray = gray[self.rgb_roi[1]:min(gray.shape[0], self.rgb_roi[1]+self.rgb_roi[3]),
                        self.rgb_roi[0]:min(gray.shape[1], self.rgb_roi[0]+self.rgb_roi[2])]
            #gray = gray[0:502, 122:584]
            #print('shape : ', gray.shape)
            #print("roi : ", self.roi[0], self.roi[1], self.roi[2], self.roi[3])

        rects = self.detector(gray, 1)

        if len(rects) > 0:
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            # face bounding box
            self.rgb_bbox = (max(0, self.rgb_roi[0]+x-round(w*0.02)), max(0, self.rgb_roi[1]+y-round(h*0.3)),
                             round(w), round(h*1.3))
            # difference between roi and bounding box
            self.rgb_diff_roi_bbox = (self.rgb_roi[0] - self.rgb_bbox[0], self.rgb_roi[1] - self.rgb_bbox[1],
                                      self.rgb_roi[2] - self.rgb_bbox[2], self.rgb_roi[3] - self.rgb_bbox[3])
            # difference between the face bounding box in t frame and the face bounding box in t-1 frane
            if self.rgb_pre_bbox[3] != 0:
                self.rgb_diff_bbox = [self.rgb_bbox[0] - self.rgb_pre_bbox[0], self.rgb_bbox[1] - self.rgb_pre_bbox[1],
                                      self.rgb_bbox[2] - self.rgb_pre_bbox[2], self.rgb_bbox[3] - self.rgb_pre_bbox[3]]
                for i in range(4):
                    if abs(self.rgb_diff_bbox[i]) < 10:
                        self.rgb_diff_bbox[i] = 0
            #if self.rgb_diff_bbox[0] != 0 or self.rgb_diff_bbox[1] != 0 or self.rgb_diff_bbox[2] != 0 or self.rgb_diff_bbox[3] != 0:
            #    print("diff bbox : ", self.rgb_diff_bbox)
            #    cv2.waitKey(0)
            self.rgb_pre_bbox = self.rgb_bbox
            #self.bbox = (max(0, 122+x-round(w*0.02)), max(0, y-round(h*0.3)),
            #             round(w), round(h*1.3))
            face_frame = frame[self.rgb_bbox[1]:min(frame.shape[0], self.rgb_bbox[1]+self.rgb_bbox[3]),
                               self.rgb_bbox[0]:min(frame.shape[1], self.rgb_bbox[0]+self.rgb_bbox[2])]
            #print("rgb bbox : ", self.rgb_bbox)
            #face_frame = frame[y-int(0.2*h):y+int(h*1.3), x-int(w*0.02):x+int(w)]
            # bbox = (x, y, w, h)
            #cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #print("detect : ", x, y, w, h)
            #print("adjust bbox : ", self.rgb_bbox)
            #print("diff roi bbox : ", self.rgb_diff_roi_bbox)
            #self.roi = (max(0, self.roi[0]+x-round(0.5*w)), max(0, self.roi[1]+y-round(0.5*h)),
            #            round(w*2), round(h*2))
            self.rgb_roi = (max(0, self.rgb_bbox[0]+self.rgb_diff_roi_bbox[0]+self.rgb_diff_bbox[0]),
                            max(0, self.rgb_bbox[1]+self.rgb_diff_roi_bbox[1]+self.rgb_diff_bbox[1]),
                            round(w*self.w_weights), round(h*self.h_weights))
            #print("roi bbox : ", self.rgb_roi)
            self.tracker1.init(frame, self.rgb_bbox)


        else:
            
            self.rgb_not_found_count += 1
            print("didn't found face %d times" % self.rgb_not_found_count)
            self.error = 1
            if self.rgb_bbox[0] > 0:
                face_frame = frame[self.rgb_bbox[1]:min(frame.shape[0], self.rgb_bbox[1]+self.rgb_bbox[3]),
                                   self.rgb_bbox[0]:min(frame.shape[1], self.rgb_bbox[0]+self.rgb_bbox[2])]
            else:
                return None
        #print(" ")

        #if not self.rgb_first_detect:
        #    cv2.imshow('gray roi', gray)
        #    cv2.waitKey(0)

        return face_frame

    def face_detect_gray(self, frame):
        if frame is None:
            print("No frame to do face detection")
            return
        #rects = self.detector(frame, 1)
        #if len(rects) > 0:
        #    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        #    self.h2 = y
        #    face_frame = frame[y-int(0.25*h):y+int(h*1.25), x-int(w*0.15):x+int(w*1.15)]
        #    bbox = (x-int(w*0.1), y-int(h*0.1), int(w*1.2), int(h*1.1))
        #    # bbox = (x, y, w, h)
        #    self.tracker2.init(frame, bbox)
        #else:
        #    return None
        roi_img = frame.copy()
        if self.nir_first_detect:
            rects = self.detector(roi_img, 1)
            if len(rects) > 0:
                (x, y, w, h) = face_utils.rect_to_bb(rects[0])
                self.nir_roi = (max(0, x-round(0.5*w)), max(0, y-round(0.5*h)), round(w*2), round(h*2))
                self.nir_first_detect = False

        if not self.nir_first_detect:
            roi_img = frame[self.nir_roi[1]:min(frame.shape[0], self.nir_roi[1]+self.nir_roi[3]),
                            self.nir_roi[0]:min(frame.shape[1], self.nir_roi[0]+self.nir_roi[2])]

        rects = self.detector(roi_img, 1)

        if len(rects) > 0:
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            self.nir_bbox = (max(0, self.nir_roi[0]+x-round(w*0.02)), max(0, self.nir_roi[1]+y-round(h*0.3)),
                             round(w), round(h*1.3))
            self.nir_diff_roi_bbox = (self.nir_roi[0] - self.nir_bbox[0], self.nir_roi[1] - self.nir_bbox[1],
                                      self.nir_roi[2] - self.nir_bbox[2], self.nir_roi[3] - self.nir_bbox[3])
            if self.nir_pre_bbox[3] != 0:
                self.nir_diff_bbox = (self.nir_bbox[0] - self.nir_pre_bbox[0], self.nir_bbox[1] - self.nir_pre_bbox[1],
                                      self.nir_bbox[2] - self.nir_pre_bbox[2], self.nir_bbox[3] - self.nir_pre_bbox[3])
            self.nir_pre_bbox = self.nir_bbox
            #self.bbox = (max(0, 122+x-round(w*0.02)), max(0, y-round(h*0.3)),
            #             round(w), round(h*1.3))
            face_frame = frame[self.nir_bbox[1]:min(frame.shape[0], self.nir_bbox[1]+self.nir_bbox[3]),
                               self.nir_bbox[0]:min(frame.shape[1], self.nir_bbox[0]+self.nir_bbox[2])]
            #face_frame = frame[y-int(0.2*h):y+int(h*1.3), x-int(w*0.02):x+int(w)]
            # bbox = (x, y, w, h)
            #cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #print("detect : ", x, y, w, h)
            #print("adjust bbox : ", self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3])
            #print("diff bbox : ", self.diff_bbox[0], self.diff_bbox[1], self.diff_bbox[2], self.diff_bbox[3])
            #print("diff roi bbox : ", self.diff_roi_bbox[0], self.diff_roi_bbox[1], self.diff_roi_bbox[2], self.diff_roi_bbox[3])
            #self.roi = (max(0, self.roi[0]+x-round(0.5*w)), max(0, self.roi[1]+y-round(0.5*h)),
            #            round(w*2), round(h*2))
            self.nir_roi = (max(0, self.nir_bbox[0]+self.nir_diff_roi_bbox[0]+self.nir_diff_bbox[0]),
                            max(0, self.nir_bbox[1]+self.nir_diff_roi_bbox[1]+self.nir_diff_bbox[1]),
                            round(w*2), round(h*2))
            self.tracker2.init(frame, self.nir_bbox)

        else:
            self.nir_not_found_count += 1
            print("didn't found face %d times" % self.nir_not_found_count)
            if self.nir_bbox[0] > 0:
                face_frame = frame[self.nir_bbox[1]:min(frame.shape[0], self.nir_bbox[1]+self.nir_bbox[3]),
                                   self.nir_bbox[0]:min(frame.shape[1], self.nir_bbox[0]+self.nir_bbox[2])]
            else:
                return None
        return face_frame
        
        

    def face_track_rgb(self, frame):
        if frame is None:
            print("No frame to do face tracking")
            return

        ok, bbox = self.tracker1.update(frame)
        if ok:
            frame = frame[bbox[1]:bbox[1]+int(bbox[3]), bbox[0]:bbox[0]+bbox[2]]
            rect = dlib.rectangle(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
            rects = dlib.rectangles()
            rects.append(rect)
            #print("track : ", bbox[0], bbox[1], bbox[2], bbox[3])
        else:
            print("Update Tracker failure")
            return
        return frame, rects

    def face_track_gray(self, frame):
        if frame is None:
            print("No frame to do face tracking")
            return
        ok, bbox = self.tracker2.update(frame)
        if ok:
            frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            rect = dlib.rectangle(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
            rects = dlib.rectangles()
            rects.append(rect)
        else:
            print("Update Tracker failure")
            return
        return frame, rects

    def detect_landmark(self, frame, face_rects):
        for k, d in enumerate(face_rects):
            shape = self.landmark_predictor(frame, d)
        #    for i in range(68):
        #        cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (255, 0, 0), -1, 3)
        #        cv2.putText(frame, str(i), (shape.part(i).x,shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        #cv2.imshow("landmark", frame)
        return shape

    # get left eye, right eye and mouse's landmark
    def get_key_landmark(self, shape):
        return [shape.part(17), shape.part(26), shape.part(8)]

    def face_patch(self, img_frame, face_rect):
        shape = self.detect_landmark(img_frame, face_rect)
        key_landmark = self.get_key_landmark(shape)

        leftEye = key_landmark[0]
        rightEye = key_landmark[1]
        mouse = key_landmark[2]
        top_y = min(leftEye.y, rightEye.y)
        bottom_y = mouse.y
        left_x = leftEye.x
        right_x = rightEye.x
        width = right_x - left_x
        height = bottom_y - top_y
        face_patch_rect = [left_x, top_y, width, height]
        #face_patch = img_frame[top_y:top_y+height, left_x:left_x+width]
        face_patch, black_point = self.remove_eyes_mouse(img_frame, shape, face_patch_rect)

        return face_patch, black_point

    def pt_to_rect(self, key_landmark):
        left_pt = key_landmark[0].x
        top_pt = key_landmark[1].y
        right_pt = key_landmark[2].x
        bottom_pt = key_landmark[3].y
        w = right_pt - left_pt
        h = bottom_pt - top_pt
        return [left_pt, top_pt, w, h]

    def remove_eyes_mouse(self, img_frame, shape, face_patch_rect):
        left_eye = [shape.part(36), shape.part(37), shape.part(39), shape.part(40)]
        right_eye = [shape.part(42), shape.part(43), shape.part(45), shape.part(46)]
        mouse = [shape.part(48), shape.part(52), shape.part(54), shape.part(57)]

        left_eye_rect = self.pt_to_rect(left_eye)
        right_eye_rect = self.pt_to_rect(right_eye)
        mouse_rect = self.pt_to_rect(mouse)

        output_img = img_frame.copy()


        output_img[left_eye_rect[1]:left_eye_rect[1]+left_eye_rect[3],
             left_eye_rect[0]:left_eye_rect[0]+left_eye_rect[2]] = '0'
        output_img[right_eye_rect[1]:right_eye_rect[1]+right_eye_rect[3],
             right_eye_rect[0]:right_eye_rect[0]+right_eye_rect[2]] = '0'
        output_img[mouse_rect[1]:mouse_rect[1]+mouse_rect[3],
             mouse_rect[0]:mouse_rect[0]+mouse_rect[2]] = '0'

        face_patch = output_img[face_patch_rect[1]:face_patch_rect[1]+face_patch_rect[3],
                               face_patch_rect[0]:face_patch_rect[0]+face_patch_rect[2]]

        black_point = left_eye_rect[2]*left_eye_rect[3] + right_eye_rect[2]*right_eye_rect[3] + mouse_rect[2]*mouse_rect[3]

        #cv2.imshow("face patch ", face_patch)
        #cv2.waitKey(1)
        return face_patch, black_point
