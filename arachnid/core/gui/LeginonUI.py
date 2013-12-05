'''
.. Created on Dec 4, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from pyui.LeginonUI import Ui_Form
from util.qt4_loader import QtGui,QtCore,qtSlot
from ..metadata import leginondb
import logging
import sys, os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class Widget(QtGui.QWidget): 
    ''' Display Controls for the LeginonUI
    '''
    
    def __init__(self, parent=None):
        "Initialize LeginonUI widget"
        
        QtGui.QWidget.__init__(self, parent)
        
        root = logging.getLogger()
        while len(root.handlers) > 0: root.removeHandler(root.handlers[0])
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(message)s'))
        h.setLevel(logging.INFO)
        root.addHandler(h)
        
        # Build window
        _logger.info("\rBuilding main window ...")
        self.ui = Ui_Form()
        self.ui.setupUi(self)
    
    @qtSlot()
    def on_changeUserPushButton_clicked(self):
        '''
        '''
        
        self.ui.loginStackedWidget.setCurrentIndex(1)
    
    @qtSlot()
    def on_loginPushButton_clicked(self):
        '''
        '''
        
        self.ui.loginStackedWidget.setCurrentIndex(0)
    
    @qtSlot(int)
    def on_loginStackedWidget_currentChanged(self, index):
        '''
        '''
        
        if index == 0: self.queryDatabase()
    
    def queryDatabase(self):
        '''
        '''
        
        username = self.ui.usernameLineEdit.text()
        password = self.ui.passwordLineEdit.text()
        leginonDB = self.ui.leginonDBLineEdit.text()
        projectDB = self.ui.projectDBLineEdit.text()
        alternteUser = self.ui.alternateUserLineEdit.text()
        try:
            sessions = leginondb.projects_for_user(username, password, leginonDB, projectDB, alternteUser)
        except:
            _logger.exception("Error accessing project")
            self.ui.loginStackedWidget.setCurrentIndex(1)
        else:
            self.ui.projectTableView.setModel(LeginonDBModel(sessions))
    
    def showEvent(self, evt):
        '''Window close event triggered - save project and global settings 
        
        :Parameters:
            
        evt : QCloseEvent
              Event for to close the main window
        '''
        
        # Load the settings
        _logger.info("\rLoading settings ...")
        self.readSettings()
        if self.ui.leginonDBLineEdit.text() == "":
            self.ui.loginStackedWidget.setCurrentIndex(1)
        else:
            self.ui.loginStackedWidget.setCurrentIndex(0)
            self.queryDatabase()
        QtGui.QWidget.showEvent(self, evt)
        
    def closeEvent(self, evt):
        '''Window close event triggered - save project and global settings 
        
        :Parameters:
            
        evt : QCloseEvent
              Event for to close the main window
        '''
        
        _logger.info("\rSaving settings ...")
        self.writeSettings()
        QtGui.QMainWindow.closeEvent(self, evt)
    
    def readSettings(self):
        '''
        '''
        
        settings = QtCore.QSettings()
        settings.beginGroup("LeginonUI")
        self.ui.leginonDBLineEdit.setText(settings.value('leginonDB'))
        self.ui.projectDBLineEdit.setText(settings.value('projectDB'))
        self.ui.usernameLineEdit.setText(settings.value('username'))
        self.ui.passwordLineEdit.setText(settings.value('password'))
        self.ui.alternateUserLineEdit.setText(settings.value('alternate-user'))
        settings.endGroup()
        
    def writeSettings(self):
        '''
        '''
        
        settings = QtCore.QSettings()
        settings.beginGroup("LeginonUI")
        settings.setValue('leginonDB', self.ui.leginonDBLineEdit.text())
        settings.setValue('projectDB', self.ui.projectDBLineEdit.text())
        settings.setValue('username', self.ui.usernameLineEdit.text())
        settings.setValue('password', self.ui.passwordLineEdit.text())
        settings.setValue('alternate-user', self.ui.alternateUserLineEdit.text())
        settings.endGroup()

class LeginonDBModel(QtCore.QAbstractTableModel):#QAbstractItemModel):
    '''
    '''

    def __init__(self, sessions, parent=None):
        '''
        '''
        
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = []
        self._header=['Session', 'Project', 'Images', 'Voltage', 'Pixel Size', 'Magnification', 'CS']
        
        for session in sessions[:3]:
            project=session.projects[0]
            if len(session.imagedata) == 0: continue
            #apix = res*(10.0**4.0)/xmag - microns
            voltage = session.scope.voltage/1000
            cs = session.scope.instrument.cs*1e3
            pixel_size = session.camera.pixel_size
            
            img = leginondb.find_exposures(session).all()
            
            magnification = img[0].scope.magnification
            self._data.append( (session.name, project.name, len(img), voltage, pixel_size, magnification, cs))
    
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        '''Returns the data for the given role and section in the header with the specified orientation
        
        :Parameters:
        
        section : QModelIndex
                  Index of a model item
        orientation : QModelIndex
                      Index of a model item
        role : QModelIndex
               Role of data to return
        
        :Returns:
        
        val : ItemFlags
              Flags for item at index
        '''
        
        if (orientation, role) == (QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole):
            return self._header[section]
        return None
    
    def data(self, index, role = QtCore.Qt.DisplayRole):
        ''' Returns the data stored under the given role for the item referred to by the index
        
        :Parameters:
        
        index : QModelIndex
                Index of a model item
        role : enum
               Data role
        
        :Returns:
        
        val : object
              Data object
        '''
        
        if not index.isValid(): return None
        if role == QtCore.Qt.DisplayRole:
            return str(self._data[index.row()][index.column()])
        return None
    
    def rowCount(self, parent = QtCore.QModelIndex()):
        ''' Get the number of rows for the parent item
        
        :Parameters:
    
        parent : QModelIndex
                 Parent of item
        
        :Returns:
        
        val : int
              Number of rows
        '''
        
        return len(self._data) if self._data is not None else 0
    
    def columnCount(self, parent = QtCore.QModelIndex()):
        ''' Get the number of columns - 2
        
        :Parameters:
    
        parent : QModelIndex
                 Parent of item
        
        :Returns:
        
        val : int
              Return value: 2
        '''
        
        return len(self._data[0]) if self._data is not None and len(self._data) > 0 else 0
    
