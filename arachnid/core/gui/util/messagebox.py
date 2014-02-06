'''
.. Created on Dec 17, 2013
.. codeauthor:: robertlanglois
'''
from qt4_loader import QtGui #,QtCore
import sys
import traceback

def error_message(parent, msg, details=""):
    '''
    '''
    
    #msgBox = QtGui.QMessageBox(QtGui.QMessageBox.Critical, u'Error', QtGui.QMessageBox.Ok, parent)
    msgBox = QtGui.QMessageBox(parent)
    msgBox.setIcon(QtGui.QMessageBox.Critical)
    msgBox.setWindowTitle('Error')
    msgBox.addButton(QtGui.QMessageBox.Ok)
    msgBox.setText(msg)
    if details != "":
        msgBox.setDetailedText(details)
    msgBox.exec_()
    
def exception_message(parent, msg, exception=None):
    '''
    '''
    
    if exception is None:
        exc_type, exc_value = sys.exc_info()[:2]
    else:
        exc_type, exc_value = exception.exc_type, exception.exc_value
    error_message(parent, msg, traceback.format_exception_only(exc_type, exc_value)[0])