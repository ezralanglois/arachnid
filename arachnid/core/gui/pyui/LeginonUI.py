# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/robertlanglois/workspace/arachnida/src/arachnid/core/gui/pyui/LeginonUI.ui'
#
# Created: Tue Jan 14 13:39:15 2014
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from ..util.qt4_loader import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(643, 426)
        self.verticalLayout = QtGui.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtGui.QWidget(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.loginStackedWidget = QtGui.QStackedWidget(self.widget)
        self.loginStackedWidget.setObjectName("loginStackedWidget")
        self.welcomePage = QtGui.QWidget()
        self.welcomePage.setObjectName("welcomePage")
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.welcomePage)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget_2 = QtGui.QWidget(self.welcomePage)
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.widget_2)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label = QtGui.QLabel(self.widget_2)
        font = QtGui.QFont()
        font.setFamily("Baskerville")
        font.setPointSize(24)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        self.verticalLayout_4.addWidget(self.label)
        self.widget_3 = QtGui.QWidget(self.widget_2)
        self.widget_3.setObjectName("widget_3")
        self.gridLayout_2 = QtGui.QGridLayout(self.widget_3)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_6 = QtGui.QLabel(self.widget_3)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 1, 1, 1, 1)
        self.entryLimitSpinBox = QtGui.QSpinBox(self.widget_3)
        self.entryLimitSpinBox.setMinimum(1)
        self.entryLimitSpinBox.setMaximum(100000)
        self.entryLimitSpinBox.setObjectName("entryLimitSpinBox")
        self.gridLayout_2.addWidget(self.entryLimitSpinBox, 1, 2, 1, 1)
        self.reloadTableToolButton = QtGui.QToolButton(self.widget_3)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/mini/mini/arrow_refresh.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.reloadTableToolButton.setIcon(icon)
        self.reloadTableToolButton.setObjectName("reloadTableToolButton")
        self.gridLayout_2.addWidget(self.reloadTableToolButton, 1, 3, 1, 1)
        self.changeUserPushButton = QtGui.QPushButton(self.widget_3)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/mini/mini/user_edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.changeUserPushButton.setIcon(icon1)
        self.changeUserPushButton.setObjectName("changeUserPushButton")
        self.gridLayout_2.addWidget(self.changeUserPushButton, 0, 2, 1, 1)
        self.userInformationToolButton = QtGui.QToolButton(self.widget_3)
        self.userInformationToolButton.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/mini/mini/information.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.userInformationToolButton.setIcon(icon2)
        self.userInformationToolButton.setObjectName("userInformationToolButton")
        self.gridLayout_2.addWidget(self.userInformationToolButton, 0, 3, 1, 1)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.widget_3)
        self.verticalLayout_3.addWidget(self.widget_2)
        self.projectTableView = QtGui.QTableView(self.welcomePage)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.projectTableView.sizePolicy().hasHeightForWidth())
        self.projectTableView.setSizePolicy(sizePolicy)
        self.projectTableView.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.projectTableView.setProperty("showDropIndicator", False)
        self.projectTableView.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        self.projectTableView.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.projectTableView.setObjectName("projectTableView")
        self.verticalLayout_3.addWidget(self.projectTableView)
        self.loginStackedWidget.addWidget(self.welcomePage)
        self.loginPage = QtGui.QWidget()
        self.loginPage.setObjectName("loginPage")
        self.gridLayout = QtGui.QGridLayout(self.loginPage)
        self.gridLayout.setContentsMargins(0, 0, 0, 3)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtGui.QLabel(self.loginPage)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.leginonHostnameLineEdit = QtGui.QLineEdit(self.loginPage)
        self.leginonHostnameLineEdit.setObjectName("leginonHostnameLineEdit")
        self.gridLayout.addWidget(self.leginonHostnameLineEdit, 1, 1, 1, 1)
        self.label_3 = QtGui.QLabel(self.loginPage)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 7, 0, 1, 1)
        self.usernameLineEdit = QtGui.QLineEdit(self.loginPage)
        self.usernameLineEdit.setObjectName("usernameLineEdit")
        self.gridLayout.addWidget(self.usernameLineEdit, 7, 1, 1, 1)
        self.label_5 = QtGui.QLabel(self.loginPage)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.projectHostnameLineEdit = QtGui.QLineEdit(self.loginPage)
        self.projectHostnameLineEdit.setObjectName("projectHostnameLineEdit")
        self.gridLayout.addWidget(self.projectHostnameLineEdit, 3, 1, 1, 1)
        self.passwordLineEdit = QtGui.QLineEdit(self.loginPage)
        self.passwordLineEdit.setEchoMode(QtGui.QLineEdit.PasswordEchoOnEdit)
        self.passwordLineEdit.setObjectName("passwordLineEdit")
        self.gridLayout.addWidget(self.passwordLineEdit, 8, 1, 1, 1)
        self.label_4 = QtGui.QLabel(self.loginPage)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 8, 0, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 12, 1, 1, 1)
        self.loginPushButton = QtGui.QPushButton(self.loginPage)
        self.loginPushButton.setObjectName("loginPushButton")
        self.gridLayout.addWidget(self.loginPushButton, 10, 0, 1, 1)
        self.leginonDBNameLineEdit = QtGui.QLineEdit(self.loginPage)
        self.leginonDBNameLineEdit.setObjectName("leginonDBNameLineEdit")
        self.gridLayout.addWidget(self.leginonDBNameLineEdit, 1, 2, 1, 1)
        self.projectDBNameLineEdit = QtGui.QLineEdit(self.loginPage)
        self.projectDBNameLineEdit.setObjectName("projectDBNameLineEdit")
        self.gridLayout.addWidget(self.projectDBNameLineEdit, 3, 2, 1, 1)
        self.label_8 = QtGui.QLabel(self.loginPage)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 1, 1, 1)
        self.label_7 = QtGui.QLabel(self.loginPage)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 0, 2, 1, 1)
        self.dbUsernameLineEdit = QtGui.QLineEdit(self.loginPage)
        self.dbUsernameLineEdit.setEchoMode(QtGui.QLineEdit.Password)
        self.dbUsernameLineEdit.setObjectName("dbUsernameLineEdit")
        self.gridLayout.addWidget(self.dbUsernameLineEdit, 7, 2, 1, 1)
        self.dbPasswordLineEdit = QtGui.QLineEdit(self.loginPage)
        self.dbPasswordLineEdit.setEchoMode(QtGui.QLineEdit.Password)
        self.dbPasswordLineEdit.setObjectName("dbPasswordLineEdit")
        self.gridLayout.addWidget(self.dbPasswordLineEdit, 8, 2, 1, 1)
        self.label_9 = QtGui.QLabel(self.loginPage)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 4, 2, 1, 1)
        self.label_10 = QtGui.QLabel(self.loginPage)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 4, 1, 1, 1)
        self.label_11 = QtGui.QLabel(self.loginPage)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setMaximumSize(QtCore.QSize(250, 16777215))
        self.label_11.setWordWrap(True)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 10, 2, 1, 1)
        self.loginStackedWidget.addWidget(self.loginPage)
        self.verticalLayout_2.addWidget(self.loginStackedWidget)
        self.verticalLayout.addWidget(self.widget)

        self.retranslateUi(Form)
        self.loginStackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.leginonHostnameLineEdit, self.projectHostnameLineEdit)
        Form.setTabOrder(self.projectHostnameLineEdit, self.usernameLineEdit)
        Form.setTabOrder(self.usernameLineEdit, self.passwordLineEdit)
        Form.setTabOrder(self.passwordLineEdit, self.loginPushButton)
        Form.setTabOrder(self.loginPushButton, self.changeUserPushButton)
        Form.setTabOrder(self.changeUserPushButton, self.entryLimitSpinBox)
        Form.setTabOrder(self.entryLimitSpinBox, self.reloadTableToolButton)
        Form.setTabOrder(self.reloadTableToolButton, self.projectTableView)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Form", "Welcome", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("Form", "Show last", None, QtGui.QApplication.UnicodeUTF8))
        self.entryLimitSpinBox.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Number of the most recent sessions to retrieve</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/information.png\" /> Only those sessions apart of projects you belong with be listed!</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.reloadTableToolButton.setToolTip(QtGui.QApplication.translate("Form", "Reload the most recent sessions from the database", None, QtGui.QApplication.UnicodeUTF8))
        self.reloadTableToolButton.setText(QtGui.QApplication.translate("Form", "...", None, QtGui.QApplication.UnicodeUTF8))
        self.changeUserPushButton.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/user_edit.png\" /> Change the Leginon database information including User Login.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/information.png\" /> Only the most recent session is displayed. Increasing the number in the Show last box and then clicking the reload <img src=\":/mini/mini/arrow_refresh.png\" /> button will load additional recent sessions.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/information.png\" /> You can select multiple sessions for processing.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/exclamation.png\" /> Only the pixel size on the images is checked, do not combine samples of different macromolecules or under different imaging conditions.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.changeUserPushButton.setText(QtGui.QApplication.translate("Form", "Change User...", None, QtGui.QApplication.UnicodeUTF8))
        self.projectTableView.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You may selection multiple sessions.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/exclamation.png\" /> Only the pixel size on the images is checked, do not combine samples of different macromolecules or under different imaging conditions.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Form", "Leginon DB", None, QtGui.QApplication.UnicodeUTF8))
        self.leginonHostnameLineEdit.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Hostname or IP for the Leginon Database</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/asterisk_orange.png\" /> Note that the Database Information will be supplied by your System Administrator</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.leginonHostnameLineEdit.setText(QtGui.QApplication.translate("Form", "bb02frank15.cpmc.columbia.edu", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Form", "Username", None, QtGui.QApplication.UnicodeUTF8))
        self.usernameLineEdit.setToolTip(QtGui.QApplication.translate("Form", "Your Leginon Username", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("Form", "Project DB", None, QtGui.QApplication.UnicodeUTF8))
        self.projectHostnameLineEdit.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Hostname or IP for the Leginon Project Database</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/asterisk_orange.png\" /> Note that the Database Information will be supplied by your System Administrator</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.projectHostnameLineEdit.setText(QtGui.QApplication.translate("Form", "bb02frank15.cpmc.columbia.edu", None, QtGui.QApplication.UnicodeUTF8))
        self.passwordLineEdit.setToolTip(QtGui.QApplication.translate("Form", "Your Leginon password (will be encrypted)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Form", "Password", None, QtGui.QApplication.UnicodeUTF8))
        self.loginPushButton.setText(QtGui.QApplication.translate("Form", "Login", None, QtGui.QApplication.UnicodeUTF8))
        self.leginonDBNameLineEdit.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Name of the Leginon Database</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/asterisk_orange.png\" /> Note that the Database Information will be supplied by your System Administrator</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.leginonDBNameLineEdit.setText(QtGui.QApplication.translate("Form", "leginondb", None, QtGui.QApplication.UnicodeUTF8))
        self.projectDBNameLineEdit.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Name of the Leginon Project Database</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/asterisk_orange.png\" /> Note that the Database Information will be supplied by your System Administrator</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.projectDBNameLineEdit.setText(QtGui.QApplication.translate("Form", "projectdb", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Host name <img src=\":/mini/mini/asterisk_orange.png\" /></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Database Name <img src=\":/mini/mini/asterisk_orange.png\" /></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.dbUsernameLineEdit.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Username for the database</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/asterisk_orange.png\" /> Note that the Database Information will be supplied by your System Administrator</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.dbUsernameLineEdit.setText(QtGui.QApplication.translate("Form", "robertl", None, QtGui.QApplication.UnicodeUTF8))
        self.dbPasswordLineEdit.setToolTip(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Password for the database</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/asterisk_orange.png\" /> Note that the Database Information will be supplied by your System Administrator</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.dbPasswordLineEdit.setText(QtGui.QApplication.translate("Form", "guest", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setText(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Database Credentials <img src=\":/mini/mini/asterisk_orange.png\" /></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label_10.setText(QtGui.QApplication.translate("Form", "Leginon Credentials", None, QtGui.QApplication.UnicodeUTF8))
        self.label_11.setToolTip(QtGui.QApplication.translate("Form", "Hostname or IP for the Leginon Database\n"
"", None, QtGui.QApplication.UnicodeUTF8))
        self.label_11.setText(QtGui.QApplication.translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Lucida Grande\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/mini/mini/asterisk_orange.png\" /> Note that the Database Information will be supplied by your System Administrator</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))

from ..icons import icons_rc;icons_rc;
