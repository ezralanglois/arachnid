# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/robertlanglois/workspace/arachnida/src/arachnid/core/gui/pyui/ReferenceUI.ui'
#
# Created: Wed Mar  5 12:28:36 2014
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from ..util.qt4_loader import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(541, 489)
        self.horizontalLayout_2 = QtGui.QHBoxLayout(Form)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.referenceTabWidget = QtGui.QTabWidget(Form)
        self.referenceTabWidget.setObjectName("referenceTabWidget")
        self.tab = QtGui.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout = QtGui.QHBoxLayout(self.tab)
        self.horizontalLayout.setContentsMargins(3, 0, 3, 3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget_3 = QtGui.QWidget(self.tab)
        self.widget_3.setMinimumSize(QtCore.QSize(200, 200))
        self.widget_3.setObjectName("widget_3")
        self.gridLayout_3 = QtGui.QGridLayout(self.widget_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.referenceLineEdit = QtGui.QLineEdit(self.widget_3)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.referenceLineEdit.sizePolicy().hasHeightForWidth())
        self.referenceLineEdit.setSizePolicy(sizePolicy)
        self.referenceLineEdit.setObjectName("referenceLineEdit")
        self.gridLayout_3.addWidget(self.referenceLineEdit, 0, 0, 1, 1)
        self.referenceFilePushButton = QtGui.QPushButton(self.widget_3)
        self.referenceFilePushButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/mini/mini/folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.referenceFilePushButton.setIcon(icon)
        self.referenceFilePushButton.setAutoDefault(False)
        self.referenceFilePushButton.setFlat(True)
        self.referenceFilePushButton.setObjectName("referenceFilePushButton")
        self.gridLayout_3.addWidget(self.referenceFilePushButton, 0, 1, 1, 1)
        self.widget_4 = QtGui.QWidget(self.widget_3)
        self.widget_4.setObjectName("widget_4")
        self.gridLayout_4 = QtGui.QGridLayout(self.widget_4)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_12 = QtGui.QLabel(self.widget_4)
        self.label_12.setObjectName("label_12")
        self.gridLayout_4.addWidget(self.label_12, 0, 0, 1, 1)
        self.referenceDepthLabel = QtGui.QLabel(self.widget_4)
        self.referenceDepthLabel.setObjectName("referenceDepthLabel")
        self.gridLayout_4.addWidget(self.referenceDepthLabel, 0, 1, 1, 1)
        self.label_14 = QtGui.QLabel(self.widget_4)
        self.label_14.setObjectName("label_14")
        self.gridLayout_4.addWidget(self.label_14, 1, 0, 1, 1)
        self.label_15 = QtGui.QLabel(self.widget_4)
        self.label_15.setObjectName("label_15")
        self.gridLayout_4.addWidget(self.label_15, 2, 0, 1, 1)
        self.referenceWidthLabel = QtGui.QLabel(self.widget_4)
        self.referenceWidthLabel.setObjectName("referenceWidthLabel")
        self.gridLayout_4.addWidget(self.referenceWidthLabel, 1, 1, 1, 1)
        self.referenceHeightLabel = QtGui.QLabel(self.widget_4)
        self.referenceHeightLabel.setObjectName("referenceHeightLabel")
        self.gridLayout_4.addWidget(self.referenceHeightLabel, 2, 1, 1, 1)
        self.label_18 = QtGui.QLabel(self.widget_4)
        self.label_18.setObjectName("label_18")
        self.gridLayout_4.addWidget(self.label_18, 3, 0, 1, 1)
        self.referencePixelSizeDoubleSpinBox = QtGui.QDoubleSpinBox(self.widget_4)
        self.referencePixelSizeDoubleSpinBox.setDecimals(3)
        self.referencePixelSizeDoubleSpinBox.setObjectName("referencePixelSizeDoubleSpinBox")
        self.gridLayout_4.addWidget(self.referencePixelSizeDoubleSpinBox, 3, 1, 1, 1)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem, 4, 1, 1, 1)
        self.gridLayout_3.addWidget(self.widget_4, 1, 0, 1, 1)
        self.label_2 = QtGui.QLabel(self.widget_3)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 2, 0, 1, 1)
        self.horizontalLayout.addWidget(self.widget_3)
        self.referenceTabWidget.addTab(self.tab, icon, "")
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.tab_2)
        self.verticalLayout_6.setContentsMargins(3, 0, 3, 3)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.emdbCannedListView = QtGui.QListView(self.tab_2)
        self.emdbCannedListView.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.emdbCannedListView.setViewMode(QtGui.QListView.IconMode)
        self.emdbCannedListView.setObjectName("emdbCannedListView")
        self.verticalLayout_6.addWidget(self.emdbCannedListView)
        self.widget_8 = QtGui.QWidget(self.tab_2)
        self.widget_8.setObjectName("widget_8")
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.widget_8)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.emdbNumberLineEdit = QtGui.QLineEdit(self.widget_8)
        self.emdbNumberLineEdit.setObjectName("emdbNumberLineEdit")
        self.horizontalLayout_3.addWidget(self.emdbNumberLineEdit)
        self.emdbDownloadPushButton = QtGui.QPushButton(self.widget_8)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/mini/mini/bullet_arrow_down.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.emdbDownloadPushButton.setIcon(icon1)
        self.emdbDownloadPushButton.setAutoDefault(False)
        self.emdbDownloadPushButton.setFlat(False)
        self.emdbDownloadPushButton.setObjectName("emdbDownloadPushButton")
        self.horizontalLayout_3.addWidget(self.emdbDownloadPushButton)
        self.openURLToolButton = QtGui.QToolButton(self.widget_8)
        self.openURLToolButton.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/mini/mini/world_link.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.openURLToolButton.setIcon(icon2)
        self.openURLToolButton.setObjectName("openURLToolButton")
        self.horizontalLayout_3.addWidget(self.openURLToolButton)
        self.downloadInformationToolButton = QtGui.QToolButton(self.widget_8)
        self.downloadInformationToolButton.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/mini/mini/information.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.downloadInformationToolButton.setIcon(icon3)
        self.downloadInformationToolButton.setObjectName("downloadInformationToolButton")
        self.horizontalLayout_3.addWidget(self.downloadInformationToolButton)
        self.verticalLayout_6.addWidget(self.widget_8)
        self.mapInfoPlainTextEdit = QtGui.QPlainTextEdit(self.tab_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mapInfoPlainTextEdit.sizePolicy().hasHeightForWidth())
        self.mapInfoPlainTextEdit.setSizePolicy(sizePolicy)
        self.mapInfoPlainTextEdit.setMinimumSize(QtCore.QSize(0, 0))
        self.mapInfoPlainTextEdit.setMaximumSize(QtCore.QSize(16777215, 50))
        self.mapInfoPlainTextEdit.setPlainText("")
        self.mapInfoPlainTextEdit.setOverwriteMode(True)
        self.mapInfoPlainTextEdit.setObjectName("mapInfoPlainTextEdit")
        self.verticalLayout_6.addWidget(self.mapInfoPlainTextEdit)
        self.label_3 = QtGui.QLabel(self.tab_2)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_6.addWidget(self.label_3)
        self.referenceTabWidget.addTab(self.tab_2, icon2, "")
        self.horizontalLayout_2.addWidget(self.referenceTabWidget)

        self.retranslateUi(Form)
        self.referenceTabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.referenceTabWidget.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Double click on map icon above or enter code below, then click <img src=\":/mini/mini/bullet_arrow_down.png\" />Download.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.referenceLineEdit.setToolTip(QtGui.QApplication.translate("Form", "File path to the local volume (may have been downloaded by the last control)", None, QtGui.QApplication.UnicodeUTF8))
        self.referenceFilePushButton.setToolTip(QtGui.QApplication.translate("Form", "Open a file dialog to choose a local volume", None, QtGui.QApplication.UnicodeUTF8))
        self.label_12.setText(QtGui.QApplication.translate("Form", "Depth", None, QtGui.QApplication.UnicodeUTF8))
        self.referenceDepthLabel.setText(QtGui.QApplication.translate("Form", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.label_14.setText(QtGui.QApplication.translate("Form", "Width", None, QtGui.QApplication.UnicodeUTF8))
        self.label_15.setText(QtGui.QApplication.translate("Form", "Height", None, QtGui.QApplication.UnicodeUTF8))
        self.referenceWidthLabel.setText(QtGui.QApplication.translate("Form", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.referenceHeightLabel.setText(QtGui.QApplication.translate("Form", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.label_18.setText(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Map Pixel Size<img src=\":/mini/mini/error.png\" /></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.referencePixelSizeDoubleSpinBox.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Pixel size of the volume choosen.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/exclamation.png\" /> You must set manually if the pixel size cannot be extracted from the header of the file. Common when the volume is in SPIDER format.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/error.png\" /> You must set manually if the pixel size cannot be extracted from the header of the file. Common when the volume is in SPIDER format.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.referenceTabWidget.setTabText(self.referenceTabWidget.indexOf(self.tab), QtGui.QApplication.translate("Form", "Local File", None, QtGui.QApplication.UnicodeUTF8))
        self.emdbCannedListView.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/bullet_blue.png\" />Use this control if you need to retrieve a volume from the EMDB (otherwise go to Local File at the top)</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/bullet_blue.png\" />You may choose a &quot;canned&quot; reference available in the list by double clicking on the icon. Then click the Download Button. This will take some time.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/bullet_blue.png\" />Alternatively, You can obtain more information on the choosen volume by clicking the <img src=\":/mini/mini/world_link.png\" />. This will open the appropriate webpage in an extern browser.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/bullet_blue.png\" />You can also enter an EMDB accession number in the text field to the left of the Download button.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.emdbNumberLineEdit.setToolTip(QtGui.QApplication.translate("Form", "Enter the EMDB Accession Number", None, QtGui.QApplication.UnicodeUTF8))
        self.emdbDownloadPushButton.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/bullet_blue.png\" />Use this control if you need to retrieve a volume from the EMDB (otherwise go to Local File at the top)</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/bullet_blue.png\" />You may choose a &quot;canned&quot; reference available in the list by double clicking on the icon. Then click the Download Button. This will take some time.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/bullet_blue.png\" />Alternatively, You can obtain more information on the choosen volume by clicking the <img src=\":/mini/mini/world_link.png\" />. This will open the appropriate webpage in an extern browser.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/bullet_blue.png\" />You can also enter an EMDB accession number in the text field to the left of the Download button.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.emdbDownloadPushButton.setText(QtGui.QApplication.translate("Form", "Download", None, QtGui.QApplication.UnicodeUTF8))
        self.openURLToolButton.setToolTip(QtGui.QApplication.translate("Form", "Open the Webpage corresponding to the EMDB accession code", None, QtGui.QApplication.UnicodeUTF8))
        self.mapInfoPlainTextEdit.setToolTip(QtGui.QApplication.translate("Form", "Citation for the chosen reference (unless you manually entered the code!)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">References curated and icons designed by Nam Ho</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.referenceTabWidget.setTabText(self.referenceTabWidget.indexOf(self.tab_2), QtGui.QApplication.translate("Form", "EMDB", None, QtGui.QApplication.UnicodeUTF8))

from ..icons import icons_rc;icons_rc;
