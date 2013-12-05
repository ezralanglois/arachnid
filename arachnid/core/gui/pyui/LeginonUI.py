# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/robertlanglois/workspace/arachnida/src/arachnid/core/gui/pyui/LeginonUI.ui'
#
# Created: Wed Dec  4 12:03:49 2013
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from ..util.qt4_loader import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(396, 421)
        self.verticalLayout = QtGui.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.loginStackedWidget = QtGui.QStackedWidget(Form)
        self.loginStackedWidget.setObjectName("loginStackedWidget")
        self.welcomePage = QtGui.QWidget()
        self.welcomePage.setObjectName("welcomePage")
        self.formLayout = QtGui.QFormLayout(self.welcomePage)
        self.formLayout.setObjectName("formLayout")
        self.label = QtGui.QLabel(self.welcomePage)
        font = QtGui.QFont()
        font.setFamily("Baskerville")
        font.setPointSize(24)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label)
        self.alternateUserLineEdit = QtGui.QLineEdit(self.welcomePage)
        self.alternateUserLineEdit.setReadOnly(False)
        self.alternateUserLineEdit.setObjectName("alternateUserLineEdit")
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.alternateUserLineEdit)
        self.changeUserPushButton = QtGui.QPushButton(self.welcomePage)
        self.changeUserPushButton.setObjectName("changeUserPushButton")
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.changeUserPushButton)
        self.loginStackedWidget.addWidget(self.welcomePage)
        self.loginPage = QtGui.QWidget()
        self.loginPage.setObjectName("loginPage")
        self.formLayout_2 = QtGui.QFormLayout(self.loginPage)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_2 = QtGui.QLabel(self.loginPage)
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_2)
        self.leginonDBLineEdit = QtGui.QLineEdit(self.loginPage)
        self.leginonDBLineEdit.setObjectName("leginonDBLineEdit")
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.FieldRole, self.leginonDBLineEdit)
        self.label_3 = QtGui.QLabel(self.loginPage)
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_3)
        self.usernameLineEdit = QtGui.QLineEdit(self.loginPage)
        self.usernameLineEdit.setObjectName("usernameLineEdit")
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.FieldRole, self.usernameLineEdit)
        self.label_4 = QtGui.QLabel(self.loginPage)
        self.label_4.setObjectName("label_4")
        self.formLayout_2.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_4)
        self.passwordLineEdit = QtGui.QLineEdit(self.loginPage)
        self.passwordLineEdit.setEchoMode(QtGui.QLineEdit.PasswordEchoOnEdit)
        self.passwordLineEdit.setObjectName("passwordLineEdit")
        self.formLayout_2.setWidget(3, QtGui.QFormLayout.FieldRole, self.passwordLineEdit)
        self.projectDBLineEdit = QtGui.QLineEdit(self.loginPage)
        self.projectDBLineEdit.setObjectName("projectDBLineEdit")
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.FieldRole, self.projectDBLineEdit)
        self.loginPushButton = QtGui.QPushButton(self.loginPage)
        self.loginPushButton.setObjectName("loginPushButton")
        self.formLayout_2.setWidget(4, QtGui.QFormLayout.FieldRole, self.loginPushButton)
        self.loginStackedWidget.addWidget(self.loginPage)
        self.verticalLayout.addWidget(self.loginStackedWidget)
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
        self.projectTableView = QtGui.QTableView(self.widget)
        self.projectTableView.setObjectName("projectTableView")
        self.verticalLayout_2.addWidget(self.projectTableView)
        self.verticalLayout.addWidget(self.widget)

        self.retranslateUi(Form)
        self.loginStackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Form", "Welcome", None, QtGui.QApplication.UnicodeUTF8))
        self.changeUserPushButton.setText(QtGui.QApplication.translate("Form", "Settings...", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Form", "URI", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Form", "Username", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Form", "Password", None, QtGui.QApplication.UnicodeUTF8))
        self.loginPushButton.setText(QtGui.QApplication.translate("Form", "Login", None, QtGui.QApplication.UnicodeUTF8))

