# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/robertlanglois/workspace/arachnida/src/arachnid/core/gui/dialogs/pyui/PropertyDialog.ui'
#
# Created: Mon Sep  3 16:32:04 2012
#      by: PyQt4 UI code generator 4.8.2
#
# WARNING! All changes made in this file will be lost!

from arachnid.core.gui.util.qt4_loader import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_PropertyDialog(object):
    def setupUi(self, PropertyDialog):
        PropertyDialog.setObjectName(_fromUtf8("PropertyDialog"))
        PropertyDialog.resize(400, 300)
        PropertyDialog.setInputMethodHints(QtCore.Qt.ImhNone)
        PropertyDialog.setModal(False)
        self.verticalLayout = QtGui.QVBoxLayout(PropertyDialog)
        self.verticalLayout.setMargin(1)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.tabWidget = QtGui.QTabWidget(PropertyDialog)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.tab)
        self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.propertyTreeView = QtGui.QTreeView(self.tab)
        self.propertyTreeView.setFrameShape(QtGui.QFrame.NoFrame)
        self.propertyTreeView.setAutoScroll(False)
        self.propertyTreeView.setEditTriggers(QtGui.QAbstractItemView.CurrentChanged|QtGui.QAbstractItemView.EditKeyPressed|QtGui.QAbstractItemView.SelectedClicked)
        self.propertyTreeView.setAlternatingRowColors(True)
        self.propertyTreeView.setIndentation(15)
        self.propertyTreeView.setUniformRowHeights(True)
        self.propertyTreeView.setObjectName(_fromUtf8("propertyTreeView"))
        self.propertyTreeView.header().setCascadingSectionResizes(True)
        self.horizontalLayout.addWidget(self.propertyTreeView)
        self.tabWidget.addTab(self.tab, _fromUtf8(""))
        self.verticalLayout.addWidget(self.tabWidget)

        self.retranslateUi(PropertyDialog)
        QtCore.QMetaObject.connectSlotsByName(PropertyDialog)

    def retranslateUi(self, PropertyDialog):
        PropertyDialog.setWindowTitle(QtGui.QApplication.translate("PropertyDialog", "Properties", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QtGui.QApplication.translate("PropertyDialog", "Tab", None, QtGui.QApplication.UnicodeUTF8))

