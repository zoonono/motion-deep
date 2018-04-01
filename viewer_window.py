# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'viewer_window.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 10, 781, 511))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setGeometry(QtCore.QRect(21, 530, 101, 22))
        self.spinBox.setObjectName("spinBox")
        self.spinBox_2 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_2.setGeometry(QtCore.QRect(140, 530, 101, 22))
        self.spinBox_2.setObjectName("spinBox_2")
        self.radioButton_axial = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_axial.setGeometry(QtCore.QRect(620, 530, 51, 17))
        self.radioButton_axial.setChecked(True)
        self.radioButton_axial.setObjectName("radioButton_axial")
        self.radioButton_sagittal = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_sagittal.setGeometry(QtCore.QRect(670, 530, 61, 17))
        self.radioButton_sagittal.setObjectName("radioButton_sagittal")
        self.radioButton_coronal = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_coronal.setGeometry(QtCore.QRect(730, 530, 61, 17))
        self.radioButton_coronal.setObjectName("radioButton_coronal")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.spinBox.setPrefix(_translate("MainWindow", "Slice: "))
        self.spinBox_2.setPrefix(_translate("MainWindow", "Example: "))
        self.radioButton_axial.setText(_translate("MainWindow", "Axial"))
        self.radioButton_sagittal.setText(_translate("MainWindow", "Sagittal"))
        self.radioButton_coronal.setText(_translate("MainWindow", "Coronal"))

