# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/robertlanglois/workspace/arachnida/src/arachnid/core/gui/dialogs/pyui/WorkflowDialog.ui'
#
# Created: Mon Sep  3 16:32:04 2012
#      by: PyQt4 UI code generator 4.8.2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_WorkflowDialog(object):
    def setupUi(self, WorkflowDialog):
        WorkflowDialog.setObjectName(_fromUtf8("WorkflowDialog"))
        WorkflowDialog.resize(349, 302)
        self.verticalLayout = QtGui.QVBoxLayout(WorkflowDialog)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.groupBox_5 = QtGui.QGroupBox(WorkflowDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_5.sizePolicy().hasHeightForWidth())
        self.groupBox_5.setSizePolicy(sizePolicy)
        self.groupBox_5.setFlat(True)
        self.groupBox_5.setObjectName(_fromUtf8("groupBox_5"))
        self.verticalLayout_12 = QtGui.QVBoxLayout(self.groupBox_5)
        self.verticalLayout_12.setMargin(0)
        self.verticalLayout_12.setObjectName(_fromUtf8("verticalLayout_12"))
        self.operatorList = QtGui.QListWidget(self.groupBox_5)
        self.operatorList.setObjectName(_fromUtf8("operatorList"))
        self.verticalLayout_12.addWidget(self.operatorList)
        self.verticalLayout.addWidget(self.groupBox_5)
        self.widget_4 = QtGui.QWidget(WorkflowDialog)
        self.widget_4.setObjectName(_fromUtf8("widget_4"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.widget_4)
        self.horizontalLayout_2.setMargin(0)
        self.horizontalLayout_2.setMargin(0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.addButton = QtGui.QPushButton(self.widget_4)
        self.addButton.setText(_fromUtf8(""))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/mini/mini/add.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.addButton.setIcon(icon)
        self.addButton.setFlat(True)
        self.addButton.setObjectName(_fromUtf8("addButton"))
        self.horizontalLayout_2.addWidget(self.addButton)
        self.removeButton = QtGui.QPushButton(self.widget_4)
        self.removeButton.setText(_fromUtf8(""))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/mini/mini/delete.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.removeButton.setIcon(icon1)
        self.removeButton.setFlat(True)
        self.removeButton.setObjectName(_fromUtf8("removeButton"))
        self.horizontalLayout_2.addWidget(self.removeButton)
        self.downButton = QtGui.QPushButton(self.widget_4)
        self.downButton.setText(_fromUtf8(""))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8(":/mini/mini/arrow_down.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.downButton.setIcon(icon2)
        self.downButton.setFlat(True)
        self.downButton.setObjectName(_fromUtf8("downButton"))
        self.horizontalLayout_2.addWidget(self.downButton)
        self.upButton = QtGui.QPushButton(self.widget_4)
        self.upButton.setEnabled(True)
        self.upButton.setText(_fromUtf8(""))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8(":/mini/mini/arrow_up.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.upButton.setIcon(icon3)
        self.upButton.setFlat(True)
        self.upButton.setObjectName(_fromUtf8("upButton"))
        self.horizontalLayout_2.addWidget(self.upButton)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout.addWidget(self.widget_4)
        self.groupBox_4 = QtGui.QGroupBox(WorkflowDialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy)
        self.groupBox_4.setMinimumSize(QtCore.QSize(0, 0))
        self.groupBox_4.setBaseSize(QtCore.QSize(0, 0))
        self.groupBox_4.setFlat(True)
        self.groupBox_4.setObjectName(_fromUtf8("groupBox_4"))
        self.verticalLayout_11 = QtGui.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_11.setMargin(0)
        self.verticalLayout_11.setObjectName(_fromUtf8("verticalLayout_11"))
        self.workflowList = QtGui.QListWidget(self.groupBox_4)
        self.workflowList.setObjectName(_fromUtf8("workflowList"))
        self.verticalLayout_11.addWidget(self.workflowList)
        self.widget = QtGui.QWidget(self.groupBox_4)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.widget)
        self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.exitButton = QtGui.QPushButton(self.widget)
        self.exitButton.setObjectName(_fromUtf8("exitButton"))
        self.horizontalLayout.addWidget(self.exitButton)
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.verticalLayout_11.addWidget(self.widget)
        self.verticalLayout.addWidget(self.groupBox_4)

        self.retranslateUi(WorkflowDialog)
        QtCore.QMetaObject.connectSlotsByName(WorkflowDialog)

    def retranslateUi(self, WorkflowDialog):
        WorkflowDialog.setWindowTitle(QtGui.QApplication.translate("WorkflowDialog", "WorkFlow", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_5.setTitle(QtGui.QApplication.translate("WorkflowDialog", "Operators", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_4.setTitle(QtGui.QApplication.translate("WorkflowDialog", "Workflow", None, QtGui.QApplication.UnicodeUTF8))
        self.exitButton.setText(QtGui.QApplication.translate("WorkflowDialog", "Finished", None, QtGui.QApplication.UnicodeUTF8))

from arachnid.core.gui.icons import icons_rc
icons_rc;
