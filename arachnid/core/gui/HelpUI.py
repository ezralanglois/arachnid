'''

- Settings
- Monitor
- Image viewer


.. Created on Sep 5, 2013
.. codeauthor:: robertlanglois
'''
from pyui.HelpUI import Ui_Dialog
from util.qt4_loader import QtGui, QtCore #, qtSlot,QtWebKit
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class Dialog(QtGui.QDialog): 
    ''' Automated GUI build from command line options
    '''
    
    def __init__(self, parent=None):
        "Initialize screener window"
        
        QtGui.QDialog.__init__(self, parent, QtCore.Qt.CustomizeWindowHint|QtCore.Qt.WindowTitleHint|QtCore.Qt.Dialog|QtCore.Qt.WindowCloseButtonHint)
        
        # Build window
        _logger.info("Building main window ...")
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        
    def setHTML(self, text):
        '''
        '''
        
        self.ui.textEdit.document().setHtml(text)

