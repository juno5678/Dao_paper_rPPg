import cv2
import numpy as np
import pyqtgraph as pg
import webbrowser
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from process import Process
from video_rgb_only import Video_RGB
from video_realsense_file import Video
from webcam import Webcam
from webcam import Camera_RGB
import sys
import timeit
import time
import signal
import threading
from queue import Queue
import os
import csv

class GUI(object):
    def __init__(self):
        super().__init__()
        #self.initUI()  # start the UI when run
        self.input_rgb = Video_RGB()
        self.input_realsense = Video()
        self.input_rgb_camera = Camera_RGB()
        self.input_realsense_camera = Webcam()
        #self.input = self.input_rgb  # input of the app
        self.dirname = ""
        self.all_file_name = ""
        self.add_nir_mode = False
        #self.statusBar.showMessage("Input: RGB Only", 5000)
        #self.btnOpen.setEnabled(True)
        self.process = Process()
        self.status = False  # If false, not running, if true, running
        self.camera_switch = False
        self.length = 10
        self.running = False
        self.avg_bpms = []
        self.smooth_bpms = []
        self.bpm_count = 0
        self.length = 10
        self.mode = 1
        

    def main_loop(self):

        # color_frame = None
        # nir_frame = None

        if self.add_nir_mode:
            # self.input.get_frame(color_frame, nir_frame)
            
            color_frame, nir_frame = self.input_realsense.get_frame()  # Read frames from input
        else:
            
            color_frame,nir_frame = self.input_realsense.get_frame()
        #gui_img = QImage(color_frame, color_frame.shape[1], color_frame.shape[0], color_frame.strideQImage.Format_RGB888)
        # color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow('framergb', color_frame)
        # cv2.imshow('framenir', nir_frame)
        #self.lblDisplay.setPixmap(QPixmap(gui_img))  # show frame on GUI

        if self.add_nir_mode:
            
            bpm, color_face, nir_face = self.process.run(color_frame, nir_frame)
        else:
            
            bpm, color_face = self.process.run(color_frame)  # Run the main algorithm
            
        color_face_input = color_face.copy()
        # color_face_input = cv2.resize(color_face_input, (255, 255), interpolation=cv2.INTER_CUBIC)
        color_face_img = QImage(color_face_input, color_face_input.shape[1], color_face_input.shape[0],
                                color_face_input.strides[0], QImage.Format_RGB888)
        #self.lblColor.setPixmap(QPixmap(color_face_img))  # Show color face

        if self.add_nir_mode:
            nir_face_input = nir_face.copy()
            # nir_face_input = cv2.resize(nir_face_input, (255, 255), interpolation=cv2.INTER_CUBIC)
            
            nir_face_img = QImage(nir_face_input, nir_face_input.shape[1], nir_face_input.shape[0],nir_face_input.strides[0], QImage.Format_Grayscale8)
            #self.lblNir.setPixmap(QPixmap(nir_face_img))  # Show nir face

        #self.lblHR.setText("Current heart rate: " + str(float("{:.2f}".format(bpm))) + "bpm")
        if len(self.process.bpms) > 25:
            for i in range(5, 0, -1):
                self.smooth_bpms.append(np.mean(self.process.bpms[-5 * i:-5 * (i - 1)]))

        if self.process.bpms.__len__() > 1:
            #self.lblHR2.setText("Smoothed heart rate: " + str(float("{:.2f}".format(np.mean(self.process.bpms)))) + " bpm")  # Print bpm value
            #print("bpm",self.process.bpms)
            self.avg_bpms.append(np.mean(self.process.bpms))

        if self.process.count >= self.process.buffer_size + 2:  # Second condition to stop running, this is 10 seconds
            print('Average FPS is: ' + str(self.process.count / (time.time() - self.t0)))
            # print('cost time is: ' + str((time.time() - self.t0)))
            # print("Testing finished")
            print("Result: " + str(np.mean(self.process.bpms)))
            self.status = False
            self.input_realsense.stop()

            self.FPS = self.process.count / (time.time() - self.t0)
            self.Result = np.mean(self.process.bpms)

            #writer.writerow([self.input_realsense.dirname,(self.process.count / (time.time() - self.t0)),np.mean(self.process.bpms)])
            #self.btnStart.setText("Start")
            #self.btnOpen.setEnabled(True)

        #self.key_handler()  # if not the GUI cant show anything, to make the gui refresh after the end of loop

    def open_camera(self):
        # color_frame = None
        # nir_frame = None
        self.input.start()
        if not self.camera_switch:
            self.camera_switch = True
            while self.camera_switch:
                if self.add_nir_mode:
                    # self.input.get_frame(color_frame, nir_frame)
                    color_frame, nir_frame = self.input.get_frame()  # Read frames from input
                    
                else:
                    color_frame = self.input.get_frame()
                gui_img = QImage(color_frame, color_frame.shape[1], color_frame.shape[0], color_frame.strides[0],
                                 QImage.Format_RGB888)
                self.lblDisplay.setPixmap(QPixmap(gui_img))  # show frame on GUI
                self.key_handler()  # if not the GUI cant show anything, to make the gui refresh after the end of loop
        else:
            self.camera_switch = False

    def estimate_sequence_bpm(self):
        
        self.input_realsense.start()

        self.t0 = time.time()
        # print('start time is: ' + str(self.t0))
        while self.running:
            
            self.main_loop()
            #self.signal_Plt.clear()
            #self.fft_Plt.plot(self.process.FREQUENCY[:300], self.process.PSD[:300], pen='r')  # Plot fused PSD

            #self.trend_Plt.clear()
            #self.trend_Plt.plot(self.process.test4, pen='r')  # Plot NIR's PSD

            #self.test1_Plt.clear()
            #self.test1_Plt.plot(self.process.bpms[-50:], pen='r')  # Plot each component's PSD

            #self.test3_Plt.clear()
            #self.test3_Plt.plot(self.smooth_bpms[:], pen='r')

    def reset(self):
        self.process.reset()

    def estimate_single_bpm(self):
        if not self.status:
            self.reset()
            self.status = True
            #self.btnStart.setText("Stop")
            #
            #self.lblHR2.clear()
            if not self.camera_switch:
                self.input_realsense.start()

            self.t0 = time.time()
            # print('start time is: ' + str(self.t0))
            while self.status:
                self.main_loop()
                #self.signal_Plt.clear()
                #self.signal_Plt.plot(self.process.RGB_signal_buffer[1], pen='r')  # Plot green signal

                #self.fft_Plt.clear()
                #self.fft_Plt.plot(self.process.FREQUENCY[:300], self.process.PSD[:300], pen='r')  # Plot fused PSD

                #self.trend_Plt.clear()
                #self.trend_Plt.plot(self.process.test4, pen='r')  # Plot NIR's PSD

                #self.test1_Plt.clear()
                #self.test1_Plt.plot(self.process.FREQUENCY[:300], self.process.test1[:300], pen='r')  # Plot each component's PSD

                #self.test2_Plt.clear()
                #self.test2_Plt.plot(self.process.FREQUENCY[:300], self.process.test2[:300],pen='r')  # Plot each component's PSD

                #self.test3_Plt.clear()
                #self.test3_Plt.plot(self.process.FREQUENCY[:300], self.process.test3[:300], pen='r')
                self.running = False

        elif self.status:
            self.status = False
            self.input.stop()
            self.btnStart.setText("Start")

    def run(self):
        while self.running:
            print("running")
            self.input_realsense.dirname = self.dirname
            self.process.set_mode(self.mode)
            self.process.set_length(self.length)
            if self.cbbInput.currentIndex() == 0 and self.input_realsense.dirname == "":
                print("Choose a video first")
                self.statusBar.showMessage("Choose a video first", 5000)
                return

            if self.cbbOutput.currentIndex() == 0:
                self.estimate_single_bpm()
            else:
                self.estimate_sequence_bpm()

    def dirpath(self,i):
        #self.input_realsense.dirname = self.dirname + str("\\") + self.all_file_name[i]
        self.input_realsense.dirname = self.dirname + str("\\") + i
        #self.input_realsense.start()
        #self.t0 = time.time()
        #print(self.input_realsense.dirname)
    
    def write_and_read(self):
        self.dirname = os.path.dirname('G:\\paper dataset\\')
        #print(self.dirname)
        #self.all_file_name = os.listdir(self.dirname)
        #return self.all_file_name
    
    


if __name__ == '__main__':
    test = GUI()
    data_list = []
    gt_list = []
    fnpath = "D:\\test.csv"
    read_path = "D:\\10s_path_gt.csv"
    file_csv = open(fnpath, 'w', newline='')

    with open(read_path,newline='') as csvfile:#讀取資料
        rows = csv.DictReader(csvfile)

        for row in rows:
            gt_list.append(row['GT'])
            data_list.append(row['data path'])

    writer = csv.writer(file_csv)
    writer.writerow(["filename","FPS","Result","MAE"])
    test.write_and_read()
    #num = len(test.all_file_name)
    num = len(data_list)
    for i in range(0,num):
        print(i)
        #test.dirpath(i)
        test.dirpath(data_list[i])
        test.estimate_single_bpm()
        mae = abs(test.Result-float(gt_list[i]))
        writer.writerow([test.input_realsense.dirname,test.FPS,test.Result,mae])
