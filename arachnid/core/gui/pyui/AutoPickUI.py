# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/robertlanglois/workspace/arachnida/src/arachnid/core/gui/pyui/AutoPickUI.ui'
#
# Created: Thu Mar 27 16:00:50 2014
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from ..util.qt4_loader import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtGui.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.diskDoubleSpinBox = QtGui.QDoubleSpinBox(Dialog)
        self.diskDoubleSpinBox.setMinimum(0.01)
        self.diskDoubleSpinBox.setMaximum(2.0)
        self.diskDoubleSpinBox.setSingleStep(0.1)
        self.diskDoubleSpinBox.setProperty("value", 0.6)
        self.diskDoubleSpinBox.setObjectName("diskDoubleSpinBox")
        self.gridLayout.addWidget(self.diskDoubleSpinBox, 0, 1, 1, 1)
        self.diskHorizontalSlider = QtGui.QSlider(Dialog)
        self.diskHorizontalSlider.setMinimum(1)
        self.diskHorizontalSlider.setMaximum(100)
        self.diskHorizontalSlider.setProperty("value", 50)
        self.diskHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.diskHorizontalSlider.setObjectName("diskHorizontalSlider")
        self.gridLayout.addWidget(self.diskHorizontalSlider, 0, 2, 1, 1)
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.maskDoubleSpinBox = QtGui.QDoubleSpinBox(Dialog)
        self.maskDoubleSpinBox.setMinimum(0.1)
        self.maskDoubleSpinBox.setMaximum(2.0)
        self.maskDoubleSpinBox.setSingleStep(0.1)
        self.maskDoubleSpinBox.setProperty("value", 1.0)
        self.maskDoubleSpinBox.setObjectName("maskDoubleSpinBox")
        self.gridLayout.addWidget(self.maskDoubleSpinBox, 1, 1, 1, 1)
        self.maskHorizontalSlider = QtGui.QSlider(Dialog)
        self.maskHorizontalSlider.setMinimum(1)
        self.maskHorizontalSlider.setMaximum(100)
        self.maskHorizontalSlider.setProperty("value", 50)
        self.maskHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.maskHorizontalSlider.setObjectName("maskHorizontalSlider")
        self.gridLayout.addWidget(self.maskHorizontalSlider, 1, 2, 1, 1)
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.overlapDoubleSpinBox = QtGui.QDoubleSpinBox(Dialog)
        self.overlapDoubleSpinBox.setMinimum(0.01)
        self.overlapDoubleSpinBox.setMaximum(2.0)
        self.overlapDoubleSpinBox.setSingleStep(0.1)
        self.overlapDoubleSpinBox.setProperty("value", 1.0)
        self.overlapDoubleSpinBox.setObjectName("overlapDoubleSpinBox")
        self.gridLayout.addWidget(self.overlapDoubleSpinBox, 2, 1, 1, 1)
        self.overlapHorizontalSlider = QtGui.QSlider(Dialog)
        self.overlapHorizontalSlider.setMinimum(1)
        self.overlapHorizontalSlider.setMaximum(100)
        self.overlapHorizontalSlider.setProperty("value", 50)
        self.overlapHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.overlapHorizontalSlider.setObjectName("overlapHorizontalSlider")
        self.gridLayout.addWidget(self.overlapHorizontalSlider, 2, 2, 1, 1)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 4, 1, 1, 1)
        self.runPushButton = QtGui.QPushButton(Dialog)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/mini/mini/control_play_blue.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.runPushButton.setIcon(icon)
        self.runPushButton.setObjectName("runPushButton")
        self.gridLayout.addWidget(self.runPushButton, 3, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "AutoPick Controls", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "Disk Multiplier", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Mask Multiplier", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Dialog", "Overlap Multiplier", None, QtGui.QApplication.UnicodeUTF8))
        self.runPushButton.setText(QtGui.QApplication.translate("Dialog", "Run", None, QtGui.QApplication.UnicodeUTF8))

from ..icons import icons_rc;icons_rc;
