# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/robertlanglois/workspace/arachnida/src/arachnid/core/gui/pyui/MontageViewer.ui'
#
# Created: Mon Apr 21 12:21:27 2014
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from ..util.qt4_loader import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(870, 576)
        MainWindow.setToolTip("")
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.centralHLayout = QtGui.QHBoxLayout()
        self.centralHLayout.setObjectName("centralHLayout")
        self.dockWidget = QtGui.QDockWidget(self.centralwidget)
        self.dockWidget.setMaximumSize(QtCore.QSize(524287, 524287))
        self.dockWidget.setFeatures(QtGui.QDockWidget.AllDockWidgetFeatures)
        self.dockWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents = QtGui.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.horizontalLayout = QtGui.QHBoxLayout(self.dockWidgetContents)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget = QtGui.QWidget(self.dockWidgetContents)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(0, 0))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtGui.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtGui.QTabWidget(self.widget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtGui.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.tab)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, -1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.widget_3 = QtGui.QWidget(self.tab)
        self.widget_3.setMinimumSize(QtCore.QSize(200, 200))
        self.widget_3.setObjectName("widget_3")
        self.formLayout_3 = QtGui.QFormLayout(self.widget_3)
        self.formLayout_3.setContentsMargins(3, 3, 3, 3)
        self.formLayout_3.setContentsMargins(0, 0, 0, 0)
        self.formLayout_3.setHorizontalSpacing(3)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_8 = QtGui.QLabel(self.widget_3)
        self.label_8.setText("")
        self.label_8.setPixmap(QtGui.QPixmap(":/mini/mini/zoom.png"))
        self.label_8.setObjectName("label_8")
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_8)
        self.imageZoomDoubleSpinBox = QtGui.QDoubleSpinBox(self.widget_3)
        self.imageZoomDoubleSpinBox.setMinimum(0.01)
        self.imageZoomDoubleSpinBox.setMaximum(1.0)
        self.imageZoomDoubleSpinBox.setSingleStep(0.1)
        self.imageZoomDoubleSpinBox.setProperty("value", 1.0)
        self.imageZoomDoubleSpinBox.setObjectName("imageZoomDoubleSpinBox")
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.FieldRole, self.imageZoomDoubleSpinBox)
        self.zoomSlider = QtGui.QSlider(self.widget_3)
        self.zoomSlider.setMinimum(1)
        self.zoomSlider.setMaximum(100)
        self.zoomSlider.setProperty("value", 100)
        self.zoomSlider.setOrientation(QtCore.Qt.Horizontal)
        self.zoomSlider.setTickPosition(QtGui.QSlider.NoTicks)
        self.zoomSlider.setObjectName("zoomSlider")
        self.formLayout_3.setWidget(1, QtGui.QFormLayout.FieldRole, self.zoomSlider)
        self.label_2 = QtGui.QLabel(self.widget_3)
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/mini/mini/contrast.png"))
        self.label_2.setObjectName("label_2")
        self.formLayout_3.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_2)
        self.contrastSlider = QtGui.QSlider(self.widget_3)
        self.contrastSlider.setMinimum(1)
        self.contrastSlider.setMaximum(400)
        self.contrastSlider.setPageStep(40)
        self.contrastSlider.setProperty("value", 200)
        self.contrastSlider.setOrientation(QtCore.Qt.Horizontal)
        self.contrastSlider.setTickPosition(QtGui.QSlider.NoTicks)
        self.contrastSlider.setObjectName("contrastSlider")
        self.formLayout_3.setWidget(2, QtGui.QFormLayout.FieldRole, self.contrastSlider)
        self.label_4 = QtGui.QLabel(self.widget_3)
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap(":/mini/mini/page_refresh.png"))
        self.label_4.setObjectName("label_4")
        self.formLayout_3.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_4)
        self.pageSpinBox = QtGui.QSpinBox(self.widget_3)
        self.pageSpinBox.setMinimum(1)
        self.pageSpinBox.setMaximum(99999)
        self.pageSpinBox.setObjectName("pageSpinBox")
        self.formLayout_3.setWidget(3, QtGui.QFormLayout.FieldRole, self.pageSpinBox)
        self.reloadPageButton = QtGui.QPushButton(self.widget_3)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/mini/mini/arrow_rotate_anticlockwise.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.reloadPageButton.setIcon(icon)
        self.reloadPageButton.setObjectName("reloadPageButton")
        self.formLayout_3.setWidget(4, QtGui.QFormLayout.FieldRole, self.reloadPageButton)
        self.verticalLayout_4.addWidget(self.widget_3)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.tab_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_2 = QtGui.QWidget(self.tab_2)
        self.widget_2.setObjectName("widget_2")
        self.formLayout_2 = QtGui.QFormLayout(self.widget_2)
        self.formLayout_2.setFieldGrowthPolicy(QtGui.QFormLayout.FieldsStayAtSizeHint)
        self.formLayout_2.setContentsMargins(0, 0, 0, 4)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_3 = QtGui.QLabel(self.widget_2)
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_3)
        self.label_5 = QtGui.QLabel(self.widget_2)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_5)
        self.clampDoubleSpinBox = QtGui.QDoubleSpinBox(self.widget_2)
        self.clampDoubleSpinBox.setMinimum(1.0)
        self.clampDoubleSpinBox.setSingleStep(1.0)
        self.clampDoubleSpinBox.setProperty("value", 5.0)
        self.clampDoubleSpinBox.setObjectName("clampDoubleSpinBox")
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.FieldRole, self.clampDoubleSpinBox)
        self.label = QtGui.QLabel(self.widget_2)
        self.label.setObjectName("label")
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.LabelRole, self.label)
        self.decimateSpinBox = QtGui.QDoubleSpinBox(self.widget_2)
        self.decimateSpinBox.setMinimum(1.0)
        self.decimateSpinBox.setMaximum(16.0)
        self.decimateSpinBox.setObjectName("decimateSpinBox")
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.FieldRole, self.decimateSpinBox)
        self.loadImagesPushButton = QtGui.QPushButton(self.widget_2)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/mini/mini/arrow_rotate_clockwise.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.loadImagesPushButton.setIcon(icon1)
        self.loadImagesPushButton.setObjectName("loadImagesPushButton")
        self.formLayout_2.setWidget(3, QtGui.QFormLayout.FieldRole, self.loadImagesPushButton)
        self.imageCountSpinBox = QtGui.QSpinBox(self.widget_2)
        self.imageCountSpinBox.setFrame(True)
        self.imageCountSpinBox.setMinimum(1)
        self.imageCountSpinBox.setMaximum(9999999)
        self.imageCountSpinBox.setProperty("value", 1)
        self.imageCountSpinBox.setObjectName("imageCountSpinBox")
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.FieldRole, self.imageCountSpinBox)
        self.verticalLayout_2.addWidget(self.widget_2)
        self.groupBox = QtGui.QGroupBox(self.tab_2)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setContentsMargins(0, 2, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.advancedSettingsTreeView = QtGui.QTreeView(self.groupBox)
        self.advancedSettingsTreeView.setEditTriggers(QtGui.QAbstractItemView.DoubleClicked|QtGui.QAbstractItemView.EditKeyPressed|QtGui.QAbstractItemView.SelectedClicked)
        self.advancedSettingsTreeView.setAlternatingRowColors(True)
        self.advancedSettingsTreeView.setIndentation(15)
        self.advancedSettingsTreeView.setUniformRowHeights(True)
        self.advancedSettingsTreeView.setObjectName("advancedSettingsTreeView")
        self.advancedSettingsTreeView.header().setCascadingSectionResizes(True)
        self.verticalLayout_3.addWidget(self.advancedSettingsTreeView)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.horizontalLayout.addWidget(self.widget)
        self.dockWidget.setWidget(self.dockWidgetContents)
        self.centralHLayout.addWidget(self.dockWidget)
        self.imageListView = QtGui.QListView(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imageListView.sizePolicy().hasHeightForWidth())
        self.imageListView.setSizePolicy(sizePolicy)
        self.imageListView.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.imageListView.setProperty("showDropIndicator", False)
        self.imageListView.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        self.imageListView.setMovement(QtGui.QListView.Static)
        self.imageListView.setProperty("isWrapping", True)
        self.imageListView.setSpacing(2)
        self.imageListView.setViewMode(QtGui.QListView.IconMode)
        self.imageListView.setObjectName("imageListView")
        self.centralHLayout.addWidget(self.imageListView)
        self.horizontalLayout_2.addLayout(self.centralHLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar()
        self.menubar.setGeometry(QtCore.QRect(0, 0, 870, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionOpen = QtGui.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/mini/mini/folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen.setIcon(icon2)
        self.actionOpen.setObjectName("actionOpen")
        self.actionFitToView = QtGui.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/mini/mini/arrow_in.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFitToView.setIcon(icon3)
        self.actionFitToView.setObjectName("actionFitToView")
        self.actionHome = QtGui.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/mini/mini/house.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionHome.setIcon(icon4)
        self.actionHome.setObjectName("actionHome")
        self.actionZoom = QtGui.QAction(MainWindow)
        self.actionZoom.setCheckable(True)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/mini/mini/zoom_out.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionZoom.setIcon(icon5)
        self.actionZoom.setObjectName("actionZoom")
        self.actionForward = QtGui.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/mini/mini/resultset_next.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionForward.setIcon(icon6)
        self.actionForward.setObjectName("actionForward")
        self.actionBackward = QtGui.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/mini/mini/resultset_previous.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionBackward.setIcon(icon7)
        self.actionBackward.setObjectName("actionBackward")
        self.actionSave = QtGui.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/mini/mini/disk.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave.setIcon(icon8)
        self.actionSave.setObjectName("actionSave")
        self.actionShow_Options = QtGui.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/mini/mini/database_table.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionShow_Options.setIcon(icon9)
        self.actionShow_Options.setObjectName("actionShow_Options")
        self.actionHide_Controls = QtGui.QAction(MainWindow)
        self.actionHide_Controls.setCheckable(True)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/mini/mini/application_side_list.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionHide_Controls.setIcon(icon10)
        self.actionHide_Controls.setObjectName("actionHide_Controls")
        self.actionOriginal_Size = QtGui.QAction(MainWindow)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(":/mini/mini/arrow_out.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOriginal_Size.setIcon(icon11)
        self.actionOriginal_Size.setObjectName("actionOriginal_Size")
        self.actionHelp = QtGui.QAction(MainWindow)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(":/mini/mini/help.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionHelp.setIcon(icon12)
        self.actionHelp.setObjectName("actionHelp")
        self.actionLoad_More = QtGui.QAction(MainWindow)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap(":/mini/mini/arrow_refresh.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionLoad_More.setIcon(icon13)
        self.actionLoad_More.setObjectName("actionLoad_More")
        self.actionAdvanced_Settings = QtGui.QAction(MainWindow)
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap(":/mini/mini/wrench_orange.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAdvanced_Settings.setIcon(icon14)
        self.actionAdvanced_Settings.setObjectName("actionAdvanced_Settings")
        self.actionSwap_Image = QtGui.QAction(MainWindow)
        self.actionSwap_Image.setCheckable(True)
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap(":/mini/mini/image.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon15.addPixmap(QtGui.QPixmap(":/mini/mini/cd.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionSwap_Image.setIcon(icon15)
        self.actionSwap_Image.setObjectName("actionSwap_Image")
        self.actionShow_Coordinates = QtGui.QAction(MainWindow)
        self.actionShow_Coordinates.setCheckable(True)
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap(":/mini/mini/table.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionShow_Coordinates.setIcon(icon16)
        self.actionShow_Coordinates.setObjectName("actionShow_Coordinates")
        self.actionSelection_Mode = QtGui.QAction(MainWindow)
        self.actionSelection_Mode.setCheckable(True)
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap(":/mini/mini/accept.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon17.addPixmap(QtGui.QPixmap(":/mini/mini/delete.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionSelection_Mode.setIcon(icon17)
        self.actionSelection_Mode.setObjectName("actionSelection_Mode")
        self.actionAutoPick = QtGui.QAction(MainWindow)
        icon18 = QtGui.QIcon()
        icon18.addPixmap(QtGui.QPixmap(":/mini/mini/page_white_magnify.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAutoPick.setIcon(icon18)
        self.actionAutoPick.setObjectName("actionAutoPick")
        self.actionInvert_Selection = QtGui.QAction(MainWindow)
        icon19 = QtGui.QIcon()
        icon19.addPixmap(QtGui.QPixmap(":/mini/mini/shape_square_error.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionInvert_Selection.setIcon(icon19)
        self.actionInvert_Selection.setObjectName("actionInvert_Selection")
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addAction(self.actionLoad_More)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionBackward)
        self.toolBar.addAction(self.actionForward)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionShow_Coordinates)
        self.toolBar.addAction(self.actionSwap_Image)
        self.toolBar.addSeparator()

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.imageZoomDoubleSpinBox.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/zoom.png\" /> Zoom the entire screen</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Works immediately.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.contrastSlider.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/contrast.png\" /> Change the contrast of the images displayed</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Works immediately</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.pageSpinBox.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/page_refresh.png\" /> Set the current group to display</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This does not update until the <img src=\":/mini/mini/arrow_rotate_clockwise.png\" /> Reload button is clicked.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.pageSpinBox.setSuffix(QtGui.QApplication.translate("MainWindow", " of 0", None, QtGui.QApplication.UnicodeUTF8))
        self.reloadPageButton.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/arrow_rotate_clockwise.png\" /> Reload the visible images using the above given parameters</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.reloadPageButton.setText(QtGui.QApplication.translate("MainWindow", "Reload", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QtGui.QApplication.translate("MainWindow", "General", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setToolTip(QtGui.QApplication.translate("MainWindow", "Number of images to display at once", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("MainWindow", "Viewable", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("MainWindow", "Clamp", None, QtGui.QApplication.UnicodeUTF8))
        self.clampDoubleSpinBox.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Number of standard deviations to remove outlier pixels</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Note that after you update this number, you must then hit the <img src=\":/mini/mini/arrow_rotate_clockwise.png\" /> Reload button to see a change.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("MainWindow", "Decimate", None, QtGui.QApplication.UnicodeUTF8))
        self.decimateSpinBox.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Factor to downsample images</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Note that after you update this number, you must then hit the <img src=\":/mini/mini/arrow_rotate_clockwise.png\" /> Reload button to see a change.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.loadImagesPushButton.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/arrow_rotate_clockwise.png\" /> Reload the visible images using the above given parameters</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Also updates the images for using the <span style=\" text-decoration: underline;\">Advanced Settings</span> below.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.loadImagesPushButton.setText(QtGui.QApplication.translate("MainWindow", "Reload", None, QtGui.QApplication.UnicodeUTF8))
        self.imageCountSpinBox.setToolTip(QtGui.QApplication.translate("MainWindow", "Number of images to display at once", None, QtGui.QApplication.UnicodeUTF8))
        self.imageCountSpinBox.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The number of images shown at one time.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Note that after you update this number, you must then hit the <img src=\":/mini/mini/arrow_rotate_clockwise.png\" /> Reload button to see a change.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("MainWindow", "Advanced Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.advancedSettingsTreeView.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" text-decoration: underline;\">Advanced Settings</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">These controls are experimental or advanced settings that are intended for expert users.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"> </p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QtGui.QApplication.translate("MainWindow", "Loading", None, QtGui.QApplication.UnicodeUTF8))
        self.toolBar.setWindowTitle(QtGui.QApplication.translate("MainWindow", "toolBar", None, QtGui.QApplication.UnicodeUTF8))
        self.actionOpen.setText(QtGui.QApplication.translate("MainWindow", "Open", None, QtGui.QApplication.UnicodeUTF8))
        self.actionOpen.setToolTip(QtGui.QApplication.translate("MainWindow", "Open an image or image stack", None, QtGui.QApplication.UnicodeUTF8))
        self.actionOpen.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/folder.png\" /> Open additional images files</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The supported formats include MRC, SPIDER, PNG and JPEG.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.actionOpen.setShortcut(QtGui.QApplication.translate("MainWindow", "Meta+O", None, QtGui.QApplication.UnicodeUTF8))
        self.actionFitToView.setText(QtGui.QApplication.translate("MainWindow", "Fit to View", None, QtGui.QApplication.UnicodeUTF8))
        self.actionFitToView.setToolTip(QtGui.QApplication.translate("MainWindow", "Fit all images to view", None, QtGui.QApplication.UnicodeUTF8))
        self.actionHome.setText(QtGui.QApplication.translate("MainWindow", "home", None, QtGui.QApplication.UnicodeUTF8))
        self.actionHome.setToolTip(QtGui.QApplication.translate("MainWindow", "Reset the view", None, QtGui.QApplication.UnicodeUTF8))
        self.actionZoom.setText(QtGui.QApplication.translate("MainWindow", "zoom", None, QtGui.QApplication.UnicodeUTF8))
        self.actionZoom.setToolTip(QtGui.QApplication.translate("MainWindow", "Use the cursor to zoom in on an area of the plot", None, QtGui.QApplication.UnicodeUTF8))
        self.actionForward.setText(QtGui.QApplication.translate("MainWindow", "forward", None, QtGui.QApplication.UnicodeUTF8))
        self.actionForward.setToolTip(QtGui.QApplication.translate("MainWindow", "Go to next group of images", None, QtGui.QApplication.UnicodeUTF8))
        self.actionForward.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/resultset_next.png\" />Go to next group of images</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The number of images in a group is controlled by the <span style=\" font-weight:600;\">Viewable</span> control, which is under the<span style=\" font-weight:600;\"> Loading</span> Tab of the controls widget.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You must hit <img src=\":/mini/mini/arrow_rotate_clockwise.png\" /> Reload after changing the number of Viewable to see an effect.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.actionForward.setShortcut(QtGui.QApplication.translate("MainWindow", "Space", None, QtGui.QApplication.UnicodeUTF8))
        self.actionBackward.setText(QtGui.QApplication.translate("MainWindow", "backward", None, QtGui.QApplication.UnicodeUTF8))
        self.actionBackward.setToolTip(QtGui.QApplication.translate("MainWindow", "Go to previous group of images", None, QtGui.QApplication.UnicodeUTF8))
        self.actionBackward.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/resultset_previous.png\" />Go to previous group of images</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The number of images in a group is controlled by the <span style=\" font-weight:600;\">Viewable</span> control, which is under the<span style=\" font-weight:600;\"> Loading</span> Tab of the controls widget.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You must hit <img src=\":/mini/mini/arrow_rotate_clockwise.png\" /> Reload after changing the number of Viewable to see an effect.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.actionBackward.setShortcut(QtGui.QApplication.translate("MainWindow", "Left", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave.setText(QtGui.QApplication.translate("MainWindow", "save", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave.setToolTip(QtGui.QApplication.translate("MainWindow", "Save a selection file", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/disk.png\" /> Save the selections to a file.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This action saves the selections in the SPIDER selection document format, which includes the IDs of only the selected images.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">If single images are open, then a single selection file is saved.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">If stacks of images are open, then a selection file is saved for each stack.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.actionShow_Options.setText(QtGui.QApplication.translate("MainWindow", "Show Options", None, QtGui.QApplication.UnicodeUTF8))
        self.actionShow_Options.setToolTip(QtGui.QApplication.translate("MainWindow", "Display dialog to configure plot options", None, QtGui.QApplication.UnicodeUTF8))
        self.actionHide_Controls.setText(QtGui.QApplication.translate("MainWindow", "Hide Controls", None, QtGui.QApplication.UnicodeUTF8))
        self.actionHide_Controls.setToolTip(QtGui.QApplication.translate("MainWindow", "Hide the controls", None, QtGui.QApplication.UnicodeUTF8))
        self.actionOriginal_Size.setText(QtGui.QApplication.translate("MainWindow", "Original Size", None, QtGui.QApplication.UnicodeUTF8))
        self.actionOriginal_Size.setToolTip(QtGui.QApplication.translate("MainWindow", "Zoom images to original size", None, QtGui.QApplication.UnicodeUTF8))
        self.actionHelp.setText(QtGui.QApplication.translate("MainWindow", "Help", None, QtGui.QApplication.UnicodeUTF8))
        self.actionHelp.setToolTip(QtGui.QApplication.translate("MainWindow", "Display help dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad_More.setText(QtGui.QApplication.translate("MainWindow", "Load More", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad_More.setToolTip(QtGui.QApplication.translate("MainWindow", "Scan for new images and open them", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad_More.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/arrow_refresh.png\" /> Scan for new images to load</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This action uses the currently open image filenames as a template to find new images that were generated after the program was loaded.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This allows the user to start screening before the pre-processing has finished.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.actionAdvanced_Settings.setText(QtGui.QApplication.translate("MainWindow", "Advanced Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.actionAdvanced_Settings.setToolTip(QtGui.QApplication.translate("MainWindow", "Open the Advanced Settings Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSwap_Image.setText(QtGui.QApplication.translate("MainWindow", "Swap Image", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSwap_Image.setToolTip(QtGui.QApplication.translate("MainWindow", "Swap between power spectra and micrograph", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSwap_Image.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/image.png\" />/<img src=\":/mini/mini/cd.png\" /> Swap between power spectra and micrograph</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The default behavior of ara-screen is to open the power spectra. Then under <span style=\" text-decoration: underline;\">Advanced Settings</span> of the <span style=\" font-weight:600;\">Loading</span> Tab, the decimated micrographs are set as <span style=\" font-weight:600;\">Alternate Image</span>.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This action swaps between the opened image (power spectra) and the <span style=\" font-weight:600;\">Alternate image</span> (micrograph).</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Also under Advanced Settings is the <span style=\" font-weight:600;\">Current PowerSpec</span> checkbox. This should be checked when you see a Power spectra and unchecked when you see a micrograph. It automatically alternates when this action is invoked.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The <span style=\" font-weight:600;\">Current PowerSpec</span> checkbox ensures that the <span style=\" text-decoration: underline;\">Advanced Settings</span> that apply to micrographs will only apply to micrographs and visa versa.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.actionShow_Coordinates.setText(QtGui.QApplication.translate("MainWindow", "Show Coordinates", None, QtGui.QApplication.UnicodeUTF8))
        self.actionShow_Coordinates.setToolTip(QtGui.QApplication.translate("MainWindow", "Show the coordinates of the selected particles on the micrograph images", None, QtGui.QApplication.UnicodeUTF8))
        self.actionShow_Coordinates.setWhatsThis(QtGui.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/table.png\" /> Show the coordinates of the selected particles on the micrograph images</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">When this action is selected, boxes around candidate particles will be displayed on the micrograph.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Under <span style=\" text-decoration: underline;\">Advanced Settings</span> of the <span style=\" font-weight:600;\">Loading</span> Tab, the following must be set:</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1. A coordinate file (<span style=\" font-weight:600;\">Coords</span>) must be specified (and exist for the current image).</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2. <span style=\" font-weight:600;\">Current Powerspec</span> must be unchecked </p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Additionally, the following <span style=\" text-decoration: underline;\">Advanced Settings</span> control various features of the window:</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1. <span style=\" font-weight:600;\">Window</span>: controls the window size of the box around the particles</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2. <span style=\" font-weight:600;\">Bin Window</span>: modifies both the window size and the coordinate positions</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">3. <span style=\" font-weight:600;\">Good File</span>: select a subset of the candidate particles</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">4. <span style=\" font-weight:600;\">Line Width</span>: controls the thickness of the lines of the box</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSelection_Mode.setText(QtGui.QApplication.translate("MainWindow", "Selection Mode", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSelection_Mode.setToolTip(QtGui.QApplication.translate("MainWindow", "Set the selection mode: Accept or Reject selected", None, QtGui.QApplication.UnicodeUTF8))
        self.actionAutoPick.setText(QtGui.QApplication.translate("MainWindow", "AutoPick", None, QtGui.QApplication.UnicodeUTF8))
        self.actionAutoPick.setToolTip(QtGui.QApplication.translate("MainWindow", "Launch the AutoPick Tuning Controls", None, QtGui.QApplication.UnicodeUTF8))
        self.actionInvert_Selection.setText(QtGui.QApplication.translate("MainWindow", "Invert Selection", None, QtGui.QApplication.UnicodeUTF8))
        self.actionInvert_Selection.setToolTip(QtGui.QApplication.translate("MainWindow", "Invert the current selection", None, QtGui.QApplication.UnicodeUTF8))
        self.actionInvert_Selection.setShortcut(QtGui.QApplication.translate("MainWindow", "Meta+I", None, QtGui.QApplication.UnicodeUTF8))

from ..icons import icons_rc;icons_rc;
