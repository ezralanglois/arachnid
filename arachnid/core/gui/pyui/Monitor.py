# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/robertlanglois/workspace/arachnida/src/arachnid/core/gui/pyui/Monitor.ui'
#
# Created: Mon Dec  2 13:48:25 2013
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from ..util.qt4_loader import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(414, 304)
        self.horizontalLayout = QtGui.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget_7 = QtGui.QWidget(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.widget_7.sizePolicy().hasHeightForWidth())
        self.widget_7.setSizePolicy(sizePolicy)
        self.widget_7.setObjectName("widget_7")
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.widget_7)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.jobListView = QtGui.QListView(self.widget_7)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.jobListView.sizePolicy().hasHeightForWidth())
        self.jobListView.setSizePolicy(sizePolicy)
        self.jobListView.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.jobListView.setProperty("showDropIndicator", False)
        self.jobListView.setObjectName("jobListView")
        self.horizontalLayout_2.addWidget(self.jobListView)
        self.horizontalLayout.addWidget(self.widget_7)
        self.widget_5 = QtGui.QWidget(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.widget_5.sizePolicy().hasHeightForWidth())
        self.widget_5.setSizePolicy(sizePolicy)
        self.widget_5.setObjectName("widget_5")
        self.verticalLayout = QtGui.QVBoxLayout(self.widget_5)
        self.verticalLayout.setContentsMargins(6, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.jobProgressBar = QtGui.QProgressBar(self.widget_5)
        self.jobProgressBar.setProperty("value", 0)
        self.jobProgressBar.setObjectName("jobProgressBar")
        self.verticalLayout.addWidget(self.jobProgressBar)
        self.logTextEdit = QtGui.QPlainTextEdit(self.widget_5)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logTextEdit.sizePolicy().hasHeightForWidth())
        self.logTextEdit.setSizePolicy(sizePolicy)
        self.logTextEdit.setUndoRedoEnabled(False)
        self.logTextEdit.setLineWrapMode(QtGui.QPlainTextEdit.NoWrap)
        self.logTextEdit.setReadOnly(True)
        self.logTextEdit.setObjectName("logTextEdit")
        self.verticalLayout.addWidget(self.logTextEdit)
        self.widget = QtGui.QWidget(self.widget_5)
        self.widget.setObjectName("widget")
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.widget)
        self.horizontalLayout_3.setContentsMargins(0, 6, 0, 12)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.pushButton = QtGui.QPushButton(self.widget)
        self.pushButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/mini/mini/resultset_next.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon.addPixmap(QtGui.QPixmap(":/mini/mini/stop.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.pushButton.setIcon(icon)
        self.pushButton.setCheckable(True)
        self.pushButton.setFlat(False)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_3.addWidget(self.pushButton)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.verticalLayout.addWidget(self.widget)
        self.horizontalLayout.addWidget(self.widget_5)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton.setToolTip(QtGui.QApplication.translate("Form", "Run the program", None, QtGui.QApplication.UnicodeUTF8))

from ..icons import icons_rc;icons_rc;
