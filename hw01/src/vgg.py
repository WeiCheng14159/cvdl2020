# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'vgg.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
from vggApp import vggApp
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIntValidator, QFont


class vggMainWindow(object):
    def __init__(self):
        self.app = vggApp()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(268, 415)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 310, 211, 51))
        self.label_2.setObjectName("label_2")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 211, 291))
        self.groupBox.setObjectName("groupBox")
        self.showTrainImage = QtWidgets.QPushButton(self.groupBox)
        self.showTrainImage.setGeometry(QtCore.QRect(10, 30, 191, 32))
        self.showTrainImage.setObjectName("showTrainImage")
        self.showHypterparameters = QtWidgets.QPushButton(self.groupBox)
        self.showHypterparameters.setGeometry(QtCore.QRect(10, 60, 191, 32))
        self.showHypterparameters.setObjectName("showHypterparameters")
        self.showModelStructure = QtWidgets.QPushButton(self.groupBox)
        self.showModelStructure.setGeometry(QtCore.QRect(10, 90, 191, 32))
        self.showModelStructure.setObjectName("showModelStructure")
        self.showLossAndAccuracy = QtWidgets.QPushButton(self.groupBox)
        self.showLossAndAccuracy.setGeometry(QtCore.QRect(10, 120, 191, 32))
        self.showLossAndAccuracy.setObjectName("showLossAndAccuracy")
        self.testImageIndex = QtWidgets.QLineEdit(self.groupBox)
        self.testImageIndex.setGeometry(QtCore.QRect(20, 220, 171, 21))
        self.testImageIndex.setObjectName("testImageIndex")
        self.testImageIndex.setValidator(QIntValidator(0, 9999))
        self.testImageIndex.setMaxLength(4)
        self.testImageIndex.setAlignment(Qt.AlignRight)
        self.testImageIndex.setFont(QFont("Arial", 20))

        self.inference = QtWidgets.QPushButton(self.groupBox)
        self.inference.setGeometry(QtCore.QRect(50, 250, 113, 32))
        self.inference.setObjectName("inference")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(20, 160, 171, 51))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 268, 22))
        self.menubar.setObjectName("menubar")
        self.menucvdl202_hw01 = QtWidgets.QMenu(self.menubar)
        self.menucvdl202_hw01.setObjectName("menucvdl202_hw01")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menucvdl202_hw01.menuAction())

        self.retranslateUi(MainWindow)
        self.showTrainImage.clicked.connect(self.app.show_rand_imgs)
        self.showHypterparameters.clicked.connect(self.app.show_hyperparemeter)
        self.showModelStructure.clicked.connect(self.app.show_model_structure)
        self.testImageIndex.textChanged.connect(self.app.get_inference_index)
        self.inference.clicked.connect(self.app.inference)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "Name: Cheng Wei (P76091226) \n"
                                        "CVDL2020 HW01"))
        self.groupBox.setTitle(_translate("MainWindow", "5. VGG-16"))
        self.showTrainImage.setText(_translate(
            "MainWindow", "1. Show Train Images"))
        self.showHypterparameters.setText(_translate(
            "MainWindow", "2. Show Hyperparameters"))
        self.showModelStructure.setText(_translate(
            "MainWindow", "3. Show Model Structure"))
        self.showLossAndAccuracy.setText(_translate(
            "MainWindow", "4. Show Loss and Accuracy"))
        self.inference.setText(_translate("MainWindow", "Inference"))
        self.label.setText(_translate("MainWindow", "Type in the index of test \n"
                                      " image to inference \n"
                                      " (0~9999)"))
        self.menucvdl202_hw01.setTitle(
            _translate("MainWindow", "cvdl202 hw01"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = vggMainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
