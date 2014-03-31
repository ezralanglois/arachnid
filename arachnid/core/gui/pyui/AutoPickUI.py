# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/robertlanglois/workspace/arachnida/src/arachnid/core/gui/pyui/AutoPickUI.ui'
#
# Created: Mon Mar 31 12:00:01 2014
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from ..util.qt4_loader import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(643, 426)
        self.verticalLayout = QtGui.QVBoxLayout(Form)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget_4 = QtGui.QWidget(Form)
        self.widget_4.setObjectName("widget_4")
        self.gridLayout = QtGui.QGridLayout(self.widget_4)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_7 = QtGui.QLabel(self.widget_4)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 4, 0, 1, 1)
        self.label_9 = QtGui.QLabel(self.widget_4)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 6, 0, 1, 1)
        self.diskHorizontalSlider = QtGui.QSlider(self.widget_4)
        self.diskHorizontalSlider.setMinimum(1)
        self.diskHorizontalSlider.setMaximum(100)
        self.diskHorizontalSlider.setProperty("value", 50)
        self.diskHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.diskHorizontalSlider.setObjectName("diskHorizontalSlider")
        self.gridLayout.addWidget(self.diskHorizontalSlider, 1, 2, 1, 1)
        self.label_6 = QtGui.QLabel(self.widget_4)
        self.label_6.setWhatsThis("")
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 1, 0, 1, 1)
        self.diskDoubleSpinBox = QtGui.QDoubleSpinBox(self.widget_4)
        self.diskDoubleSpinBox.setMinimum(0.01)
        self.diskDoubleSpinBox.setMaximum(2.0)
        self.diskDoubleSpinBox.setSingleStep(0.1)
        self.diskDoubleSpinBox.setProperty("value", 0.6)
        self.diskDoubleSpinBox.setObjectName("diskDoubleSpinBox")
        self.gridLayout.addWidget(self.diskDoubleSpinBox, 1, 1, 1, 1)
        self.maskDoubleSpinBox = QtGui.QDoubleSpinBox(self.widget_4)
        self.maskDoubleSpinBox.setMinimum(0.1)
        self.maskDoubleSpinBox.setMaximum(2.0)
        self.maskDoubleSpinBox.setSingleStep(0.1)
        self.maskDoubleSpinBox.setProperty("value", 1.0)
        self.maskDoubleSpinBox.setObjectName("maskDoubleSpinBox")
        self.gridLayout.addWidget(self.maskDoubleSpinBox, 4, 1, 1, 1)
        self.overlapDoubleSpinBox = QtGui.QDoubleSpinBox(self.widget_4)
        self.overlapDoubleSpinBox.setMinimum(0.01)
        self.overlapDoubleSpinBox.setMaximum(2.0)
        self.overlapDoubleSpinBox.setSingleStep(0.1)
        self.overlapDoubleSpinBox.setProperty("value", 1.0)
        self.overlapDoubleSpinBox.setObjectName("overlapDoubleSpinBox")
        self.gridLayout.addWidget(self.overlapDoubleSpinBox, 6, 1, 1, 1)
        self.overlapHorizontalSlider = QtGui.QSlider(self.widget_4)
        self.overlapHorizontalSlider.setMinimum(1)
        self.overlapHorizontalSlider.setMaximum(100)
        self.overlapHorizontalSlider.setProperty("value", 50)
        self.overlapHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.overlapHorizontalSlider.setObjectName("overlapHorizontalSlider")
        self.gridLayout.addWidget(self.overlapHorizontalSlider, 6, 2, 1, 1)
        self.maskHorizontalSlider = QtGui.QSlider(self.widget_4)
        self.maskHorizontalSlider.setMinimum(1)
        self.maskHorizontalSlider.setMaximum(100)
        self.maskHorizontalSlider.setProperty("value", 50)
        self.maskHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.maskHorizontalSlider.setObjectName("maskHorizontalSlider")
        self.gridLayout.addWidget(self.maskHorizontalSlider, 4, 2, 1, 1)
        self.verticalLayout.addWidget(self.widget_4)
        self.widget_5 = QtGui.QWidget(Form)
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.widget_5)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.runPushButton = QtGui.QPushButton(self.widget_5)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/mini/mini/control_play_blue.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.runPushButton.setIcon(icon)
        self.runPushButton.setObjectName("runPushButton")
        self.horizontalLayout_3.addWidget(self.runPushButton)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.verticalLayout.addWidget(self.widget_5)
        self.autopickHistoryTableView = QtGui.QTableView(Form)
        self.autopickHistoryTableView.setObjectName("autopickHistoryTableView")
        self.verticalLayout.addWidget(self.autopickHistoryTableView)
        self.label_7.setBuddy(self.maskDoubleSpinBox)
        self.label_9.setBuddy(self.overlapDoubleSpinBox)
        self.label_6.setBuddy(self.diskDoubleSpinBox)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setToolTip(QtGui.QApplication.translate("Form", "Mask Multiplier (--mask-mult)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("Form", "Mask ", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setToolTip(QtGui.QApplication.translate("Form", "Overlap Multiplier (--overlap-mult)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setText(QtGui.QApplication.translate("Form", "Overlap", None, QtGui.QApplication.UnicodeUTF8))
        self.diskHorizontalSlider.setToolTip(QtGui.QApplication.translate("Form", "Disk Multipler (--disk-mult)", None, QtGui.QApplication.UnicodeUTF8))
        self.diskHorizontalSlider.setWhatsThis(QtGui.QApplication.translate("Form", "Disk Multipler (--disk-mult)\n"
"\n"
"This option controls the size of the smooth disk.", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setToolTip(QtGui.QApplication.translate("Form", "Disk Multipler (--disk-mult)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("Form", "Disk", None, QtGui.QApplication.UnicodeUTF8))
        self.diskDoubleSpinBox.setToolTip(QtGui.QApplication.translate("Form", "Disk Multipler (--disk-mult)", None, QtGui.QApplication.UnicodeUTF8))
        self.diskDoubleSpinBox.setWhatsThis(QtGui.QApplication.translate("Form", "Disk Multipler (--disk-mult)\n"
"\n"
"This option controls the size of the smooth disk.", None, QtGui.QApplication.UnicodeUTF8))
        self.maskDoubleSpinBox.setToolTip(QtGui.QApplication.translate("Form", "Mask Multiplier (--mask-mult)", None, QtGui.QApplication.UnicodeUTF8))
        self.maskDoubleSpinBox.setWhatsThis(QtGui.QApplication.translate("Form", "Mask Multiplier (--mask-mult)\n"
"\n"
"This option controls the size of the particle boundary mask.", None, QtGui.QApplication.UnicodeUTF8))
        self.overlapDoubleSpinBox.setToolTip(QtGui.QApplication.translate("Form", "Overlap Multiplier (--overlap-mult)", None, QtGui.QApplication.UnicodeUTF8))
        self.overlapDoubleSpinBox.setWhatsThis(QtGui.QApplication.translate("Form", "Overlap Multiplier (--overlap-mult)\n"
"\n"
"This option controls the amount of allowed overlap between cross-correlation peaks.", None, QtGui.QApplication.UnicodeUTF8))
        self.overlapHorizontalSlider.setToolTip(QtGui.QApplication.translate("Form", "Overlap Multiplier (--overlap-mult)", None, QtGui.QApplication.UnicodeUTF8))
        self.overlapHorizontalSlider.setWhatsThis(QtGui.QApplication.translate("Form", "Overlap Multiplier (--overlap-mult)\n"
"\n"
"This option controls the amount of allowed overlap between cross-correlation peaks.", None, QtGui.QApplication.UnicodeUTF8))
        self.maskHorizontalSlider.setToolTip(QtGui.QApplication.translate("Form", "Mask Multiplier (--mask-mult)", None, QtGui.QApplication.UnicodeUTF8))
        self.maskHorizontalSlider.setWhatsThis(QtGui.QApplication.translate("Form", "Mask Multiplier (--mask-mult)\n"
"\n"
"This option controls the size of the particle boundary mask.", None, QtGui.QApplication.UnicodeUTF8))
        self.runPushButton.setToolTip(QtGui.QApplication.translate("Form", "Run AutoPicker", None, QtGui.QApplication.UnicodeUTF8))
        self.runPushButton.setWhatsThis(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" text-decoration: underline;\">Launch AutoPicker</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Clicking this button runs AutoPicker on all <span style=\" font-weight:600;\">visible and selection</span> micrographs.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.runPushButton.setText(QtGui.QApplication.translate("Form", "Run", None, QtGui.QApplication.UnicodeUTF8))
        self.autopickHistoryTableView.setToolTip(QtGui.QApplication.translate("Form", "History of each run", None, QtGui.QApplication.UnicodeUTF8))
        self.autopickHistoryTableView.setWhatsThis(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" text-decoration: underline;\">AutoPick History</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; text-decoration: underline;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This table contains the history of each AutoPicker run. It maintains the parameters used and number of micrographs tested along with the number of particles found.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Double clicking on the history item will:</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1. Display exisiting coordinates for the visible, selected micrographs</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2. Run AutoPicker on any new visible, selected micrographs</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))

from ..icons import icons_rc;icons_rc;
