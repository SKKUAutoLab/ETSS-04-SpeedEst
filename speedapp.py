# Form implementation generated from reading ui file 'SpeedApp.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
import threading
import multiprocessing

import numpy as np
from skimage import io
import cv2
from PyQt6 import QtCore, QtGui, QtWidgets
from position_confix import pairs
from SpeedEstimation_UI import speedEstimation
from SpeedEstimation_UI_Run_All import speedEstimationAll_1


NUMBER_THREAD = [1,2,3,4]
class videoProcess(QtCore.QThread):
    update_video = QtCore.pyqtSignal(dict)

    def __init__(self, filename, matrix):
        super().__init__()
        self.filename = filename
        self.matrix = matrix

    def run(self):
        speedEstimation(self.filename, self.matrix, self.update_video)
class RunAllProccess(QtCore.QThread):
    update_status = QtCore.pyqtSignal(dict)
    def __init__(self, number):
        super().__init__()
        self.thread_number = number

    def run(self):
        if self.thread_number == 1:
            speedEstimationAll_1(self.update_status)
        # elif self.thread_number == 2:
        #     speedEstimationAll_2(self.update_status)
        # elif self.thread_number == 3:
        #     speedEstimationAll_3(self.update_status)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 600)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.inputInfo = QtWidgets.QLabel(parent=self.centralwidget)
        self.inputInfo.setGeometry(QtCore.QRect(1120, 10, 380, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.inputInfo.setFont(font)
        self.inputInfo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.inputInfo.setObjectName("inputInfo")
        self.button_point1_img = QtWidgets.QPushButton(parent=self.centralwidget)
        self.button_point1_img.setGeometry(QtCore.QRect(980, 90, 80, 30))
        self.button_point1_img.setObjectName("button_point1_img")
        self.button_point2_img = QtWidgets.QPushButton(parent=self.centralwidget)
        self.button_point2_img.setGeometry(QtCore.QRect(980, 140, 80, 30))
        self.button_point2_img.setObjectName("button_point2_img")
        self.button_point3_img = QtWidgets.QPushButton(parent=self.centralwidget)
        self.button_point3_img.setGeometry(QtCore.QRect(980, 190, 80, 30))
        self.button_point3_img.setObjectName("button_point3_img")
        self.button_point4_img = QtWidgets.QPushButton(parent=self.centralwidget)
        self.button_point4_img.setGeometry(QtCore.QRect(980, 240, 80, 30))
        self.button_point4_img.setObjectName("button_point4_img")
        self.Lable4ImagePoints = QtWidgets.QLabel(parent=self.centralwidget)
        self.Lable4ImagePoints.setGeometry(QtCore.QRect(980, 50, 291, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Lable4ImagePoints.setFont(font)
        self.Lable4ImagePoints.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.Lable4ImagePoints.setObjectName("Lable4ImagePoints")
        self.Lable4WorldPoints = QtWidgets.QLabel(parent=self.centralwidget)
        self.Lable4WorldPoints.setGeometry(QtCore.QRect(1300, 50, 291, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Lable4WorldPoints.setFont(font)
        self.Lable4WorldPoints.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.Lable4WorldPoints.setObjectName("Lable4WorldPoints")
        self.button_point1_world = QtWidgets.QPushButton(parent=self.centralwidget)
        self.button_point1_world.setGeometry(QtCore.QRect(1300, 90, 80, 30))
        self.button_point1_world.setObjectName("button_point1_world")
        self.button_point2_world = QtWidgets.QPushButton(parent=self.centralwidget)
        self.button_point2_world.setGeometry(QtCore.QRect(1300, 140, 80, 30))
        self.button_point2_world.setObjectName("button_point2_world")
        self.button_point3_world = QtWidgets.QPushButton(parent=self.centralwidget)
        self.button_point3_world.setGeometry(QtCore.QRect(1300, 190, 80, 30))
        self.button_point3_world.setObjectName("button_point3_world")
        self.button_point4_world = QtWidgets.QPushButton(parent=self.centralwidget)
        self.button_point4_world.setGeometry(QtCore.QRect(1300, 240, 80, 30))
        self.button_point4_world.setObjectName("button_point4_world")
        self.LoadVideo = QtWidgets.QPushButton(parent=self.centralwidget)
        self.LoadVideo.setGeometry(QtCore.QRect(990, 330, 100, 80))
        self.LoadVideo.setObjectName("LoadVideo")
        self.Dashboad = QtWidgets.QLabel(parent=self.centralwidget)
        self.Dashboad.setGeometry(QtCore.QRect(1110, 290, 380, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.Dashboad.setFont(font)
        self.Dashboad.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.Dashboad.setObjectName("Dashboad")
        self.Capture = QtWidgets.QPushButton(parent=self.centralwidget)
        self.Capture.setGeometry(QtCore.QRect(1110, 330, 100, 80))
        self.Capture.setObjectName("Capture")
        self.Calibrate = QtWidgets.QPushButton(parent=self.centralwidget)
        self.Calibrate.setGeometry(QtCore.QRect(1230, 330, 100, 80))
        self.Calibrate.setObjectName("Calibrate")
        self.SpeedEstimation = QtWidgets.QPushButton(parent=self.centralwidget)
        self.SpeedEstimation.setGeometry(QtCore.QRect(1350, 330, 100, 80))
        self.SpeedEstimation.setObjectName("SpeedEstimation")
        self.Run_All = QtWidgets.QPushButton(parent=self.centralwidget)
        self.Run_All.setGeometry(QtCore.QRect(1470, 330, 100, 80))
        self.Run_All.setObjectName("RunAll")
        self.WindowShowVideo = QtWidgets.QLabel(parent=self.centralwidget)
        self.WindowShowVideo.setGeometry(QtCore.QRect(10, 10, 960, 540))
        self.WindowShowVideo.setMouseTracking(False)
        self.WindowShowVideo.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.WindowShowVideo.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.WindowShowVideo.setText("")
        self.WindowShowVideo.setObjectName("WindowShowVideo")
        self.label_point1_img = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_point1_img.setGeometry(QtCore.QRect(1080, 90, 191, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label_point1_img.setFont(font)
        self.label_point1_img.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_point1_img.setText("")
        self.label_point1_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_point1_img.setObjectName("label_point1_img")
        self.label_point2_img = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_point2_img.setGeometry(QtCore.QRect(1080, 140, 191, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label_point2_img.setFont(font)
        self.label_point2_img.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_point2_img.setText("")
        self.label_point2_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_point2_img.setObjectName("label_point2_img")
        self.label_point4_img = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_point4_img.setGeometry(QtCore.QRect(1080, 240, 191, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label_point4_img.setFont(font)
        self.label_point4_img.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_point4_img.setText("")
        self.label_point4_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_point4_img.setObjectName("label_point4_img")
        self.label_point3_img = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_point3_img.setGeometry(QtCore.QRect(1080, 190, 191, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label_point3_img.setFont(font)
        self.label_point3_img.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_point3_img.setText("")
        self.label_point3_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_point3_img.setObjectName("label_point3_img")
        self.label_point1_world = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_point1_world.setGeometry(QtCore.QRect(1400, 90, 191, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label_point1_world.setFont(font)
        self.label_point1_world.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_point1_world.setText("")
        self.label_point1_world.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_point1_world.setObjectName("label_point1_world")
        self.label_point2_world = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_point2_world.setGeometry(QtCore.QRect(1400, 140, 191, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label_point2_world.setFont(font)
        self.label_point2_world.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_point2_world.setText("")
        self.label_point2_world.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_point2_world.setObjectName("label_point2_world")
        self.label_point3_world = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_point3_world.setGeometry(QtCore.QRect(1400, 190, 191, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label_point3_world.setFont(font)
        self.label_point3_world.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_point3_world.setText("")
        self.label_point3_world.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_point3_world.setObjectName("label_point3_world")
        self.label_point4_world = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_point4_world.setGeometry(QtCore.QRect(1400, 240, 191, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label_point4_world.setFont(font)
        self.label_point4_world.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_point4_world.setText("")
        self.label_point4_world.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_point4_world.setObjectName("label_point4_world")

        self.label_show_guide = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_show_guide.setGeometry(QtCore.QRect(990, 420, 580, 130))
        font = QtGui.QFont()
        font.setPointSize(14)
        # font.setBold(True)
        self.label_show_guide.setFont(font)
        self.label_show_guide.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_show_guide.setText("Welcome to Speed Estimation Application!" + "\n" + "\n" + "Please load video and set points.......!!!!!!!")
        self.label_show_guide.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_show_guide.setObjectName("label_show_guide")


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 19))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        #Flag start and stop
        self.start = False
        self.current_frame = None
        self.flag_get_point = False

        # Clicked Events
        self.button_point1_img.clicked.connect(self.getPoint1)
        self.button_point2_img.clicked.connect(self.getPoint2)
        self.button_point3_img.clicked.connect(self.getPoint3)
        self.button_point4_img.clicked.connect(self.getPoint4)
        self.button_point1_world.clicked.connect(self.setPoint1World)
        self.button_point2_world.clicked.connect(self.setPoint2World)
        self.button_point3_world.clicked.connect(self.setPoint3World)
        self.button_point4_world.clicked.connect(self.setPoint4World)
        self.Calibrate.clicked.connect(self.calibrate)
        self.LoadVideo.clicked.connect(self.loadVideo)
        self.Capture.clicked.connect(self.playVideo)
        self.SpeedEstimation.clicked.connect(self.speed_Cal)
        self.Run_All.clicked.connect(self.runAll)




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Speed  Estimation App"))
        self.inputInfo.setText(_translate("MainWindow", "Input for  Camera Calibration"))
        self.button_point1_img.setText(_translate("MainWindow", "Point 1"))
        self.button_point2_img.setText(_translate("MainWindow", "Point 2"))
        self.button_point3_img.setText(_translate("MainWindow", "Point 3"))
        self.button_point4_img.setText(_translate("MainWindow", "Point 4"))
        self.Lable4ImagePoints.setText(_translate("MainWindow", "4 Points in Image"))
        self.Lable4WorldPoints.setText(_translate("MainWindow", "4 Coresponding Points in 3D Coordinate"))
        self.button_point1_world.setText(_translate("MainWindow", "Point 1"))
        self.button_point2_world.setText(_translate("MainWindow", "Point 2"))
        self.button_point3_world.setText(_translate("MainWindow", "Point 3"))
        self.button_point4_world.setText(_translate("MainWindow", "Point 4"))
        self.LoadVideo.setText(_translate("MainWindow", "Load Video"))
        self.Dashboad.setText(_translate("MainWindow", "Dashboard"))
        self.Capture.setText(_translate("MainWindow", "Capture"))
        self.Calibrate.setText(_translate("MainWindow", "Calibrate"))
        self.Run_All.setText(_translate("MainWindow", "Run All"))
        self.SpeedEstimation.setText(_translate("MainWindow", "Speed Est"))


    def loadImage(self):
        global imgName
        global rows_prop
        global cols_prop
        imgName = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "Choose Image", "", "*.jpg;;*.png;;All Files(*)")[0]
        jpg = QtGui.QPixmap(imgName).scaled(self.WindowShowVideo.width(), self.WindowShowVideo.height())
        self.WindowShowVideo.setPixmap(jpg)
        rows, cols = image_read(imgName)
        rows_prop = rows / self.WindowShowVideo.height()
        cols_prop = cols / self.WindowShowVideo.width()
        # self.WindowShowVideo.mousePressEvent = self.getPos
        # print(" event:  ", self.WindowShowVideo.mousePressEvent)

    def getPoint1(self):
        self.flag_get_point = True
        print("Call GetPoint 1")
        self.WindowShowVideo.mousePressEvent = self.getPos1

    def getPoint2(self):
        self.flag_get_point = True
        print("Call GetPoint 2")
        self.WindowShowVideo.mousePressEvent = self.getPos2

    def getPoint3(self):
        self.flag_get_point = True
        print("Call GetPoint 3")
        self.WindowShowVideo.mousePressEvent = self.getPos3

    def getPoint4(self):
        self.flag_get_point = True
        print("Call GetPoint 4")
        self.WindowShowVideo.mousePressEvent = self.getPos4

    #func get position when click on the image
    def getPos1(self, event):
        global x_1
        global y_1
        global rows_prop
        global cols_prop
        x_1 = event.pos().x() * cols_prop
        y_1 = event.pos().y() * rows_prop
        _translate = QtCore.QCoreApplication.translate
        self.label_point1_img.setText(_translate("MainWindow", "(" + str(x_1) + " , " + str(y_1) + ")"))

    def getPos2(self, event):
        global x_2
        global y_2
        global rows_prop
        global cols_prop
        x_2 = event.pos().x() * cols_prop
        y_2 = event.pos().y() * rows_prop
        _translate = QtCore.QCoreApplication.translate
        self.label_point2_img.setText(_translate("MainWindow", "(" + str(x_2) + " , " + str(y_2) + ")"))

    def getPos3(self, event):
        global x_3
        global y_3
        global rows_prop
        global cols_prop
        x_3 = event.pos().x() * cols_prop
        y_3 = event.pos().y() * rows_prop
        _translate = QtCore.QCoreApplication.translate
        self.label_point3_img.setText(_translate("MainWindow", "(" + str(x_3) + " , " + str(y_3) + ")"))

    def getPos4(self, event):
        global x_4
        global y_4
        global rows_prop
        global cols_prop
        x_4 = event.pos().x() * cols_prop
        y_4 = event.pos().y() * rows_prop
        _translate = QtCore.QCoreApplication.translate
        self.label_point4_img.setText(_translate("MainWindow", "(" + str(x_4) + " , " + str(y_4) + ")"))

    def setPoint1World(self):
        global point1_world
        point1_world, ok = QtWidgets.QInputDialog.getText(self.centralwidget, "The World Coordinate", "Please input corresponding point 1 (ex: (0,0))： ")
        if ok:
            self.label_point1_world.setText(str(point1_world))

    def setPoint2World(self):
        global point2_world
        point2_world, ok = QtWidgets.QInputDialog.getText(self.centralwidget, "The World Coordinate (m)", "Please input corresponding point 2 (ex: (0,0))： ")
        if ok:
            self.label_point2_world.setText(str(point2_world))

    def setPoint3World(self):
        global point3_world
        point3_world, ok = QtWidgets.QInputDialog.getText(self.centralwidget, "The World Coordinate (m)", "Please input corresponding point 3 (ex: (0,0))： ")
        if ok:
            self.label_point3_world.setText(str(point3_world))

    def setPoint4World(self):
        global point4_world
        point4_world, ok = QtWidgets.QInputDialog.getText(self.centralwidget, "The World Coordinate (m)", "Please input corresponding point 4 (ex: (0,0))： ")
        if ok:
            self.label_point4_world.setText(str(point4_world))
    # Func to calibrate camera
    def calibrate(self):
        global x_1
        global y_1
        global x_2
        global y_2
        global x_3
        global y_3
        global x_4
        global y_4
        global point1_world
        global point2_world
        global point3_world
        global point4_world
        global filename

        if self.flag_get_point == True:
            print("This error for get point by hand")

            #convert point  from string to array
            line_1 = self.str2int(point1_world)
            line_2 = self.str2int(point2_world)
            line_3 = self.str2int(point3_world)
            line_4 = self.str2int(point4_world)

            source_points = np.array([
                [x_1, y_1],
                [x_2, y_2],
                [x_3, y_3],
                [x_4, y_4]
            ])
            destination_points = np.array([
                line_1,
                line_2,
                line_3,
                line_4
            ])
        else:
            #get point by video
            source_points = pairs[filename.split("/")[-2]]["source_points"]
            destination_points = pairs[filename.split("/")[-2]]["destination_points"]
        matrix = getHomography(source_points, destination_points)
        self.label_show_guide.setText("Homography matrix for {}: ".format(filename.split("/")[-2]) + "\n" + "\n" + str(matrix))
        return matrix

    def str2int(self, points):
        try:
            # Remove parentheses and split the string by commas
            values = points.strip("()").split(',')
            # Convert strings to integers
            values = [int(val.strip()) for val in values]
            # Convert lists to NumPy arrays
            array = np.array(values)
            return array
        except NameError:
            print("Please insert the world point condinate")
    def loadVideo(self):
        global filename
        self.flag_get_point = False
        filename = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "Open Video", "", "*.avi;; *.mp4;;All Files(*)")[0]
        print("This is link file: ", filename)
        # print("This is file name: ", filename.split("/")[-2])
        print("This is Position in source: ", pairs[filename.split("/")[-2]])
        self.label_point1_img.setText("(" + str("{:.2f}".format(pairs[filename.split("/")[-2]]["source_points"][0][0])) + ", " +  str("{:.2f}".format(pairs[filename.split("/")[-2]]["source_points"][0][1])+  ")"))
        self.label_point2_img.setText("(" + str("{:.2f}".format(pairs[filename.split("/")[-2]]["source_points"][1][0])) + ", " +  str("{:.2f}".format(pairs[filename.split("/")[-2]]["source_points"][1][1])+  ")"))
        self.label_point3_img.setText("(" + str("{:.2f}".format(pairs[filename.split("/")[-2]]["source_points"][2][0])) + ", " +  str("{:.2f}".format(pairs[filename.split("/")[-2]]["source_points"][2][1])+  ")"))
        self.label_point4_img.setText("(" + str("{:.2f}".format(pairs[filename.split("/")[-2]]["source_points"][3][0])) + ", " +  str("{:.2f}".format(pairs[filename.split("/")[-2]]["source_points"][3][1])+  ")"))

        self.label_point1_world.setText(
            "(" + str("{:.2f}".format(pairs[filename.split("/")[-2]]["destination_points"][0][0])) + ", " + str(
                "{:.2f}".format(pairs[filename.split("/")[-2]]["destination_points"][0][1]) + ")"))
        self.label_point2_world.setText(
            "(" + str("{:.2f}".format(pairs[filename.split("/")[-2]]["destination_points"][1][0])) + ", " + str(
                "{:.2f}".format(pairs[filename.split("/")[-2]]["destination_points"][1][1]) + ")"))
        self.label_point3_world.setText(
            "(" + str("{:.2f}".format(pairs[filename.split("/")[-2]]["destination_points"][2][0])) + ", " + str(
                "{:.2f}".format(pairs[filename.split("/")[-2]]["destination_points"][2][1]) + ")"))
        self.label_point4_world.setText(
            "(" + str("{:.2f}".format(pairs[filename.split("/")[-2]]["destination_points"][3][0])) + ", " + str(
                "{:.2f}".format(pairs[filename.split("/")[-2]]["destination_points"][3][1]) + ")"))
        self.label_show_guide.setText("You can change 4 pair points if you have." + "\n" + "\n"
                                      + "Opt 1:  Please press [Capture] to get 1 frame and change points"
                                      + "\n" + "\n"
                                      + "Opt 2: If not. Please press [Calibrate] to get homography matrix"
                                      )

    # def speed_Cal(self):
    #     global filename
    #     matrix = self.calibrate()
    #     print("This is matrix in speed_cal: ", matrix)
    #     speedEstimation(filename, matrix)
    def speed_Cal(self):
        global filename
        matrix = self.calibrate()
        self.video_process = videoProcess(filename, matrix)
        self.video_process.start()
        self.video_process.update_video.connect(self.updateImage)


    def runAll(self):
        self.label_show_guide.setText("In progress ... ")
        self.run_all_video_1 = RunAllProccess(NUMBER_THREAD[0])
        # self.run_all_video_2 = RunAllProccess(NUMBER_THREAD[1])
        # self.run_all_video_3 = RunAllProccess(NUMBER_THREAD[2])
        self.run_all_video_1.start()
        # self.run_all_video_2.start()
        # self.run_all_video_3.start()
        self.run_all_video_1.update_status.connect(self.updateImage)
        # self.run_all_video_1.update_status.connect(self.updateStatus)
        self.label_show_guide.setText("======> Finish <======")


    # DEBUG using Qprocess
    # def runAll(self):
    #     # Change the start method to 'spawn'
    #     multiprocessing.set_start_method('spawn')
    #
    #     # Create processes without calling the functions
    #     process1 = multiprocessing.Process(target=speedEstimationAll_1, args=())
    #     process2 = multiprocessing.Process(target=speedEstimationAll_2, args=())
    #     process3 = multiprocessing.Process(target=speedEstimationAll_3, args=())
    #
    #     # Start the processes
    #     process1.start()
    #     process2.start()
    #     process3.start()
    #
    #     # Process IDs
    #     print("ID of process p1: {}".format(process1.pid))
    #     print("ID of process p2: {}".format(process2.pid))
    #     print("ID of process p3: {}".format(process3.pid))
    #
    #     # Wait for all processes to finish
    #     process1.join()
    #     process2.join()
    #     process3.join()



    def updateStatus(self, status_process):
        if status_process:
            self.label_show_guide.setText("======> Finish <======")




    def updateImage(self, infor):
        image = cv2.resize(infor["image"], (self.WindowShowVideo.width(), self.WindowShowVideo.height()),
                           interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format.Format_RGB888)
        self.WindowShowVideo.setPixmap(QtGui.QPixmap.fromImage(image))
        QtWidgets.QApplication.processEvents()




    def playVideo(self):
        global filename
        global rows_prop
        global cols_prop

        if self.start:
            self.start = False
            self.Capture.setText("Start")
        else:
            self.start = True
            self.Capture.setText("Stop")
        cap = cv2.VideoCapture(filename)
        while (cap.isOpened()):
            # try:
            _, frame = cap.read()
            # cv2.imshow("Image",  frame)
            # cv2.waitKey(0)
            self.current_frame = frame
            image = cv2.resize(frame, (self.WindowShowVideo.width(), self.WindowShowVideo.height()),
                               interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format.Format_RGB888)
            self.WindowShowVideo.setPixmap(QtGui.QPixmap.fromImage(image))
            QtWidgets.QApplication.processEvents()
            # except Exception as e:
            #     print("An error occurred:", e)
            #
            key = cv2.waitKey(1) & 0xFF
            if self.start == False:
                break
                print('Stop play video')

        rows = self.current_frame.shape[0]
        cols = self.current_frame.shape[1]
        rows_prop = rows / self.WindowShowVideo.height()
        cols_prop = cols / self.WindowShowVideo.width()


    # def loadVideo(self):
    #     self.player = VideoPlayer()
    #     self.player.setWindowTitle("Player")
    #     self.player.resize(900, 600)
    #     self.player.show()



def image_read(imgpath):
    image = io.imread(imgName)
    rows = image.shape[0]
    cols = image.shape[1]
    # son add
    if len(image.shape) > 2 and image.shape[2] == 4:
        # convert the image from RGBA2RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    # Son finish
    return rows, cols

def getHomography(source_points, destination_points):
    H = cv2.findHomography(source_points, destination_points)
    H = np.array(H[0]) #convert to array
    print("This is Homography Matrix: ", H.shape)
    return H

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    obj = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(obj)
    obj.show()
    sys.exit(app.exec())