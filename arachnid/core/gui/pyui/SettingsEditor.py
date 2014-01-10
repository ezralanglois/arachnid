# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/robertlanglois/workspace/arachnida/src/arachnid/core/gui/pyui/SettingsEditor.ui'
#
# Created: Wed Jan  8 11:24:43 2014
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from ..util.qt4_loader import QtCore, QtGui

class Ui_TabWidget(object):
    def setupUi(self, TabWidget):
        TabWidget.setObjectName("TabWidget")
        TabWidget.resize(400, 300)
        self.tab = QtGui.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout = QtGui.QHBoxLayout(self.tab)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.propertyTreeView = QtGui.QTreeView(self.tab)
        self.propertyTreeView.setStyleSheet("QTreeView::item {\n"
"      border: 1px solid #d9d9d9;\n"
" }")
        self.propertyTreeView.setFrameShape(QtGui.QFrame.NoFrame)
        self.propertyTreeView.setAutoScroll(False)
        self.propertyTreeView.setEditTriggers(QtGui.QAbstractItemView.CurrentChanged|QtGui.QAbstractItemView.EditKeyPressed|QtGui.QAbstractItemView.SelectedClicked)
        self.propertyTreeView.setAlternatingRowColors(False)
        self.propertyTreeView.setIndentation(15)
        self.propertyTreeView.setUniformRowHeights(True)
        self.propertyTreeView.setObjectName("propertyTreeView")
        self.propertyTreeView.header().setCascadingSectionResizes(True)
        self.horizontalLayout.addWidget(self.propertyTreeView)
        TabWidget.addTab(self.tab, "")

        self.retranslateUi(TabWidget)
        TabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(TabWidget)

    def retranslateUi(self, TabWidget):
        TabWidget.setWindowTitle(QtGui.QApplication.translate("TabWidget", "TabWidget", None, QtGui.QApplication.UnicodeUTF8))
        TabWidget.setTabText(TabWidget.indexOf(self.tab), QtGui.QApplication.translate("TabWidget", "Tab 1", None, QtGui.QApplication.UnicodeUTF8))

