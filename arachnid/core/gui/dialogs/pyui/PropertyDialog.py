# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PropertyDialog.ui'
#
# Created: Sat Jul  6 20:59:40 2013
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from arachnid.core.gui.util.qt4_loader import QtCore, QtGui

class Ui_PropertyDialog(object):
    def setupUi(self, PropertyDialog):
        PropertyDialog.setObjectName("PropertyDialog")
        PropertyDialog.resize(400, 300)
        PropertyDialog.setInputMethodHints(QtCore.Qt.ImhNone)
        PropertyDialog.setModal(False)
        self.verticalLayout = QtGui.QVBoxLayout(PropertyDialog)
        self.verticalLayout.setContentsMargins(1, 1, 1, 1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtGui.QTabWidget(PropertyDialog)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtGui.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout = QtGui.QHBoxLayout(self.tab)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.propertyTreeView = QtGui.QTreeView(self.tab)
        self.propertyTreeView.setFrameShape(QtGui.QFrame.NoFrame)
        self.propertyTreeView.setAutoScroll(False)
        self.propertyTreeView.setEditTriggers(QtGui.QAbstractItemView.CurrentChanged|QtGui.QAbstractItemView.EditKeyPressed|QtGui.QAbstractItemView.SelectedClicked)
        self.propertyTreeView.setAlternatingRowColors(True)
        self.propertyTreeView.setIndentation(15)
        self.propertyTreeView.setUniformRowHeights(True)
        self.propertyTreeView.setObjectName("propertyTreeView")
        self.propertyTreeView.header().setCascadingSectionResizes(True)
        self.horizontalLayout.addWidget(self.propertyTreeView)
        self.tabWidget.addTab(self.tab, "")
        self.verticalLayout.addWidget(self.tabWidget)

        self.retranslateUi(PropertyDialog)
        QtCore.QMetaObject.connectSlotsByName(PropertyDialog)

    def retranslateUi(self, PropertyDialog):
        PropertyDialog.setWindowTitle(QtGui.QApplication.translate("PropertyDialog", "Properties", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QtGui.QApplication.translate("PropertyDialog", "Tab", None, QtGui.QApplication.UnicodeUTF8))

