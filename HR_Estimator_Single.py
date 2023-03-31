import cv2
import numpy as np
import pyqtgraph as pg
import webbrowser
from process import Process
from video_rgb_only import Video_RGB
from video_realsense_file import Video
from image_rgb import Image_RGB
from webcam import Webcam
from webcam import Camera_RGB
import sys
import timeit
import time
import signal
import threading
from queue import Queue

class DAO_HRE():
    def __init__(self, dirname):
        self.dirname = dirname
        self.process = Process()
        self.running = False
        self.fragment_bpms = []
        self.bpm_count = 0
        self.sample_rate = 30
        self.length = 10
        self.buffer_size = self.sample_rate * self.length
        self.max_input_length = 60
        self.max_input_size = self.sample_rate * self.max_input_length
        self.process_mode = 0 # process mode , 0 : RGB , 1 : RGB + NIR, 2 : CIEa + CIEb + NIR
        self.input_mode = 1 # input mode   , 0 : rgb video , 1 : rgb image, 2 : rgb camera
                              #                3 : realsense video, 4 : realsense camera
        self.output_mode = 0  # output mode  , 0 : estimate single, 1 : estimate sequence, 2 : estimate, 3 : fragment
        self.avg_bpms = []
        self.interal_frame = 6

    def init_input(self):
        if self.input_mode == 0:
            signal_input = Video_RGB()
        elif self.input_mode == 1:
            signal_input = Image_RGB()
        elif self.input_mode == 2:
            signal_input = Camera_RGB()
        elif self.input_mode == 3:
            signal_input = Video()
        elif self.input_mode == 4:
            signal_input = Webcam()
        else:
            print("please type right input, 0 : rgb video, 1 : rgb image, 2 : rgb camera"
                  + " 3 : realsense video, 4 : realsense camera")
            return -1
        return signal_input

    def set_data_path(self, data_path):
        self.dirname = data_path

    def set_input(self, input_mode):
        self.input_mode = input_mode
        self.init_input()

    def set_output(self, output_mode):
        self.output_mode = output_mode
        self.process.output_mode = output_mode

    def set_process_mode(self, process_mode):
        self.process_mode = process_mode
        self.process.set_process_mode(process_mode)

    # set the length of time for estimate heart rate
    def set_length(self, length):
        self.process.set_length(length)
        self.length = length
        self.buffer_size = self.sample_rate * self.length

    def reset(self):
        self.process.reset()
        self.fragment_bpms = []

    def run(self):
        signal_input = self.init_input()
        signal_input.dirname = self.dirname
        signal_input.start()

        start = time.time()
        self.running = True
        frame_count = 0
        while self.running:
            # get input
            if self.input_mode == 0 or self.input_mode == 1:
                self.max_input_size = signal_input.frame_length

            if self.input_mode == 3 or self.input_mode == 4:  # has nir information
                color_frame, nir_frame = signal_input.get_frame()  # Read frames from input
            else:
                color_frame = signal_input.get_frame()

            if color_frame is None:
                self.running = False

            # process
            #print("frame count in estimator : ", frame_count)
            if self.process_mode == 0: # RGB mode
                bpm, color_face = self.process.run(color_frame)  # Run the main algorithm
            else:  # RGB + NIR or CIEa + CIEb +NIR mode
                bpm, color_face, nir_face = self.process.run(color_frame, nir_frame)

            # output
            if frame_count >= self.length * self.sample_rate:
                # output first fragment result
                if self.output_mode == 0:
                    print('current {} bpm : {:.2f} '.format(frame_count, bpm))
                    self.running = False
                    bpm_output = bpm
                # output sequence result
                elif self.output_mode == 1:
                    if frame_count >= self.max_input_size:
                        print('total {} sequence bpm : '.format(len(self.process.bpms)), self.process.bpms)
                        self.running = False
                        bpm_output = self.process.bpms
                # output fragment result
                elif self.output_mode == 2:
                    #if frame_count % int(self.buffer_size/2) == 0:
                    if frame_count % self.interal_frame == 0 or frame_count % self.interal_frame == 1:
                    #if frame_count % 6 == 0:
                        self.avg_bpms.append(bpm)
                        #self.fragment_bpms.append(bpm)
                        #if frame_count % 6 == 1:
                        if frame_count % self.interal_frame == 1:
                            avg_bpm_array = np.array(self.avg_bpms)
                            no_zero_idx = np.where(avg_bpm_array != 0)
                            if len(no_zero_idx[0]) == 2:
                                self.fragment_bpms.append(np.mean(self.avg_bpms))
                            elif len(no_zero_idx[0]) == 1:
                                self.fragment_bpms.append(avg_bpm_array[no_zero_idx][0])
                            else:
                                self.fragment_bpms.append(0)
                            #print("avg bpm : ", self.avg_bpms)
                            self.avg_bpms = []

                        if self.max_input_size % self.interal_frame != 0:
                            n = np.floor(self.max_input_size / self.interal_frame)
                        else:
                            n = np.floor(self.max_input_size / self.interal_frame) - 1

                        if frame_count >= n * self.interal_frame + 1:
                        #if frame_count >= self.max_input_size+1:
                            print('total {} fragment bpm : '.format(len(self.fragment_bpms)), self.fragment_bpms)
                            self.running = False
                            bpm_output = self.fragment_bpms

            frame_count += 1
        end = time.time()
        processing_time = end-start
        fps = frame_count/processing_time
        print("processing time : ", end-start)
        print("fps : ", fps)
        #print('signal process time ', np.mean(self.process.signal_process_t))
        #print('tracking time ', np.mean(self.process.tracking_process_t))
        return bpm_output, fps


if __name__ == '__main__':
    dirname = sys.argv[1]
    HR_Estimator = DAO_HRE(dirname)
    HR_Estimator.set_process_mode(0)
    HR_Estimator.set_length(10)
    HR_Estimator.set_output(2)
    HR_Estimator.set_input(1)  # set input 1 : image sequence
    HR_Estimator.set_data_path(dirname)
    HR_Estimator.run()
