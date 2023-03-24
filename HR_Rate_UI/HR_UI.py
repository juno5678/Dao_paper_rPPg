import pyqtgraph as pg
import webbrowser
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys


class Communicate(QObject):
    closeApp = pyqtSignal()


class GUI(QMainWindow, QThread):
    def __init__(self):
        super().__init__()
        self.initUI()  # start the UI when run
        self.dirname = ""
        self.add_nir_mode = False
        #self.statusBar.showMessage("Input: RGB Only", 5000)
        self.btnOpen.setEnabled(True)
        self.status = False  # If false, not running, if true, running

    def initUI(self):
        # set font
        font = QFont()
        font.setFamily('Adobe Gothic Std B')
        font.setBold(True)
        font.setPointSize(14)
        font.setWeight(20)

        # 測試畫面
        self.lblDisplay = QLabel(self)
        self.lblDisplay.setGeometry(120, 60, 800, 600)
        self.lblDisplay.setStyleSheet("background-color: #000000")

        # dynamic plot # Processed Signal 圖表
        self.signal_Plt = pg.PlotWidget(self)
        self.signal_Plt.setGeometry(935, 60, 625, 300)
        self.signal_Plt.setBackground('#ffffff')
        # self.signal_Plt.setOpacity(1)
        self.signal_Plt.setLabel('top', "Processed Signal")

        # Frequency Lable
        self.lblHR = QLabel(self)
        self.lblHR.setGeometry(935, 375, 305, 85)
        self.lblHR.setStyleSheet("color:white")
        self.lblHR.setFont(QFont("OCR A Std", 13, QFont.Bold))
        self.lblHR.setText("Frequency:--/--")

        # Heart Rate Lable
        self.lblHR2 = QLabel(self)
        self.lblHR2.setGeometry(1255, 375, 305, 85)
        self.lblHR2.setStyleSheet("color:white")
        self.lblHR2.setFont(QFont("OCR A Std", 13, QFont.Bold))
        self.lblHR2.setText("Heart Rate:--/--")

        # Time Lable
        self.lblTime = QLabel(self)
        self.lblTime.setGeometry(420, 60, 200, 40)
        self.lblTime.setFont(font)
        self.lblTime.setAlignment(Qt.AlignCenter)
        self.lblTime.setStyleSheet("color:#00FF00")
        self.lblTime.setText("- - . - - . - -")

        # Tip Lable
        self.lblTip = QLabel(self)
        self.lblTip.setGeometry(120, 60, 800, 600)
        self.lblTip.setFont(font)
        self.lblTip.setAlignment(Qt.AlignCenter)
        self.lblTip.setFont(QFont("OCR A Std", 18, QFont.Bold))
        self.lblTip.setStyleSheet("QLabel{color:white;}")
        self.lblTip.setText("Press Button to Start\n...")

        # CCU Logo1 button
        self.lblCCU_Logo1 = QLabel(self)
        self.lblCCU_Logo1.setGeometry(180, 5, 270, 50)
        self.lblCCU_Logo1.setStyleSheet(
            "QLabel{border-image: url(./IMG_Source/CCU_Logo.png);}")

        # CCU Logo2 button
        self.lblCCU_Logo2 = QPushButton("< Produced  by  Lab520 >", self)
        self.lblCCU_Logo2.setGeometry(640, 660, 400, 50)
        self.lblCCU_Logo2.setFont(QFont("Hanyi Senty Meadow", 16, QFont.Bold))
        self.lblCCU_Logo2.setStyleSheet("QPushButton{color: white;}")
        self.lblCCU_Logo2.setFlat(True)
        self.lblCCU_Logo2.clicked.connect(self.lblCCU_Logo2_clicked)

        # CCU Logo3 button
        self.lblCCU_Logo3 = QLabel(self)
        self.lblCCU_Logo3.setGeometry(455, 15, 270, 50)
        self.lblCCU_Logo3.setStyleSheet("color:white")
        self.lblCCU_Logo3.setFont(QFont("Adobe 宋体 Std L", 10, QFont.Bold))
        self.lblCCU_Logo3.setText("電機工程 研究所")

        # DataSource Combobox
        self.cbbInput = QComboBox(self)
        self.cbbInput.addItem(" Video (RGB)")
        self.cbbInput.addItem(" Video (RGB & NIR)")
        self.cbbInput.addItem(" Camera (RGB)")
        self.cbbInput.addItem(" Camera (RGB & NIR)")
        self.cbbInput.setCurrentIndex(0)
        self.cbbInput.setGeometry(935, 475, 625, 60)
        self.cbbInput.setFont(font)
        self.cbbInput.setStyleSheet(
            "QComboBox{color:gray;background-image: url(./IMG_Source/Button.jpg); border-radius: 10px; border: 2px groove gray;border-style: outset;}")
        self.cbbInput.activated.connect(self.selectInput)

        # Start button
        self.btnStart = QPushButton("Start \n HR Detection", self)
        self.btnStart.setGeometry(935, 550, 305, 100)
        self.btnStart.setFont(font)
        self.btnStart.setStyleSheet("QPushButton{color: gray ;border-image: url(./IMG_Source/Button.jpg); border-radius: 10px; border: 2px groove gray;border-style: outset;}"
                                    "QPushButton:hover{color: #ffffff;}"
                                    "QPushButton:pressed{color: #000000;}")

        # Open button
        self.btnOpen = QPushButton("Open \n Video File", self)
        self.btnOpen.setGeometry(1255, 550, 305, 100)
        self.btnOpen.setFont(font)
        self.btnOpen.setStyleSheet("QPushButton{color: gray ;border-image: url(./IMG_Source/Button.jpg); border-radius: 10px; border: 2px groove gray;border-style: outset;}"
                                   "QPushButton:hover{color: #ffffff;}"
                                   "QPushButton:pressed{color: #000000;}")
        self.btnOpen.clicked.connect(self.openFileDialog)

        # Pro button

        self.btnPro1 = QPushButton(self)
        self.btnPro1.setGeometry(1630, 60, 50, 200)
        self.btnPro1.setStyleSheet("QPushButton{border-image: url(./IMG_Source/Professional_Button.jpg); border-radius: 10px; border: 2px groove gray;border-style: outset;}"
                                   "QPushButton:hover{border-image: url(./IMG_Source/Button.jpg);}"
                                   "QPushButton:pressed{border-image: url(./IMG_Source/Button.jpg);}")
        self.btnPro1.clicked.connect(self.btnPro1_clicked)

        # Pro2 button
        self.btnPro2 = QPushButton("", self)
        self.btnPro2.setGeometry(1630, 260, 50, 200)
        self.btnPro2.setFont(font)

        # Pro3 button
        self.btnPro3 = QPushButton("", self)
        self.btnPro3.setGeometry(1630, 460, 50, 200)
        self.btnPro3.setFont(font)

        # Information button
        self.btnInformation = QPushButton(self)
        self.btnInformation.setGeometry(120, 10, 40, 40)
        self.btnInformation.setStyleSheet("QPushButton{border-image: url(./IMG_Source/Information_Button.png)}"
                                          "QPushButton:hover{background-color: #FFA823;}"
                                          "QPushButton:pressed{background-color: #FFA823;}")
        self.btnInformation.clicked.connect(self.btnInformation_clicked)

        # event close
        self.c = Communicate()
        self.c.closeApp.connect(self.closeEvent)

        # config main window # 視窗大小
        self.setWindowTitle("Heart Rate Monitor")
        self.setGeometry(0, 0, 1680, 720)

        self.fft_Plt = pg.PlotWidget(self)
        self.fft_Plt.setGeometry(120, 720, 273, 150)
        self.fft_Plt.setBackground('#ffffff')
        self.fft_Plt.setLabel('top', "Chosen PP")

        self.trend_Plt = pg.PlotWidget(self)
        self.trend_Plt.setGeometry(408, 720, 273, 150)
        self.trend_Plt.setBackground('#ffffff')
        self.trend_Plt.setLabel('top', "Raw Signal")

        self.test1_Plt = pg.PlotWidget(self)
        self.test1_Plt.setGeometry(696, 720, 273, 150)
        self.test1_Plt.setBackground('#ffffff')
        self.test1_Plt.setLabel('top', "Source 1 PP")

        self.test2_Plt = pg.PlotWidget(self)
        self.test2_Plt.setGeometry(984, 720, 273, 150)
        self.test2_Plt.setBackground('#ffffff')
        self.test2_Plt.setLabel('top', "Source 2 PP")

        self.test3_Plt = pg.PlotWidget(self)
        self.test3_Plt.setGeometry(1272, 720, 273, 150)
        self.test3_Plt.setBackground('#ffffff')
        self.test3_Plt.setLabel('top', "Source 3 PP")

        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(
            QPixmap("./IMG_Source/BackGround.jpg")))
        self.setPalette(palette)

        self.center()
        self.show()

    def btnPro1_clicked(self):
        self.setGeometry(0, 0, 1680, 900)
        self.center()
        self.show()

    def btnPro2_clicked(self):
        reply = QMessageBox.question(
            self, "Message", "Are you sure want to quit ?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    def btnPro3_clicked(self):
        reply = QMessageBox.question(
            self, "Message", "Are you sure want to quit ?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    def btnInformation_clicked(self):
        webbrowser.open('http://www.dsp.ee.ccu.edu.tw/wnlie/')

    def lblCCU_Logo2_clicked(self):
        webbrowser.open('http://www.dsp.ee.ccu.edu.tw/wnlie/')

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Message", "Are you sure want to quit ?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def selectInput(self):
        self.reset()
        if self.cbbInput.currentIndex() == 0:
            self.input = self.input_rgb
            self.add_nir_mode = False
            self.statusBar.showMessage("Input: RGB Only", 5000)
        else:
            self.input = self.input_realsense
            self.add_nir_mode = True
            self.statusBar.showMessage("Input: RGB + NIR", 5000)

    def openFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.dirname, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Python Files (*.py)", options=options)
        self.statusBar.showMessage("Folder name: " + self.dirname, 5000)

    def reset(self):
        self.process.reset()
        self.lblDisplay.clear()
        self.lblColor.clear()
        self.lblNir.clear()

    @QtCore.pyqtSlot()
    def main_loop(self):
        if self.add_nir_mode:
            color_frame, nir_frame = self.input.get_frame()  # Read frames from input
        else:
            color_frame = self.input.get_frame()
        gui_img = QImage(color_frame, color_frame.shape[1], color_frame.shape[0], color_frame.strides[0],
                         QImage.Format_RGB888)
        self.lblDisplay.setPixmap(QPixmap(gui_img))  # show frame on GUI

        if self.add_nir_mode:
            bpm, color_face, nir_face = self.process.run(
                color_frame, nir_frame)
        else:
            bpm, color_face = self.process.run(
                color_frame)  # Run the main algorithm

        color_face_input = color_face.copy()
        color_face_img = QImage(color_face_input, color_face_input.shape[1], color_face_input.shape[0],
                                color_face_input.strides[0], QImage.Format_RGB888)
        self.lblColor.setPixmap(QPixmap(color_face_img))  # Show color face

        if self.add_nir_mode:
            nir_face_input = nir_face.copy()
            # nir_face_input = cv2.resize(nir_face_input, (255, 255), interpolation=cv2.INTER_CUBIC)
            nir_face_img = QImage(nir_face_input, nir_face_input.shape[1], nir_face_input.shape[0],
                                  nir_face_input.strides[0], QImage.Format_Grayscale8)
            self.lblNir.setPixmap(QPixmap(nir_face_img))  # Show nir face

        self.lblHR.setText("Freq: " + str(float("{:.2f}".format(bpm))))
        if self.process.bpms.__len__() > 1:
            self.lblHR2.setText(
                "Heart rate: " + str(float("{:.2f}".format(np.mean(self.process.bpms)))) + " bpm")  # Print bpm value

        # Second condition to stop running, this is 10 seconds
        if self.process.count >= self.process.buffer_size+2:
            # print('Average FPS is: ' + str(self.process.count / (time.time() - self.t0)))
            # print("Testing finished")
            print("Result: " + str(np.mean(self.process.bpms)))
            self.status = False
            self.input.stop()
            self.btnStart.setText("Start")
            self.btnOpen.setEnabled(True)

        # if not the GUI cant show anything, to make the gui refresh after the end of loop
        self.key_handler()

    def run(self):
        self.input.dirname = self.dirname
        if self.input.dirname == "":
            print("Choose a video first")
            self.statusBar.showMessage("Choose a video first", 5000)
            return

        if not self.status:
            self.reset()
            self.status = True
            self.btnStart.setText("Stop")
            self.btnOpen.setEnabled(False)
            self.lblHR2.clear()
            self.input.start()
            self.t0 = time.time()

            while self.status:
                self.main_loop()
                self.signal_Plt.clear()
                # Plot green signal
                self.signal_Plt.plot(
                    self.process.RGB_signal_buffer[1], pen='r')

                self.fft_Plt.clear()
                # Plot fused PSD
                self.fft_Plt.plot(
                    self.process.FREQUENCY[:300], self.process.PSD[:300], pen='r')

                self.trend_Plt.clear()
                self.trend_Plt.plot(self.process.test4,
                                    pen='r')  # Plot NIR's PSD

                self.test1_Plt.clear()
                # Plot each component's PSD
                self.test1_Plt.plot(
                    self.process.FREQUENCY[:300], self.process.test1[:300], pen='r')

                self.test2_Plt.clear()
                # Plot each component's PSD
                self.test2_Plt.plot(
                    self.process.FREQUENCY[:300], self.process.test2[:300], pen='r')

                self.test3_Plt.clear()
                self.test3_Plt.plot(
                    self.process.FREQUENCY[:300], self.process.test3[:300], pen='r')

        elif self.status:
            self.status = False
            self.input.stop()
            self.btnStart.setText("Start")
            self.btnOpen.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())
