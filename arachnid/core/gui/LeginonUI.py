'''
.. Created on Dec 4, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from pyui.LeginonUI import Ui_Form
from util.qt4_loader import QtGui,QtCore,qtSlot
from model.ListTableModel import ListTableModel
from ..metadata import leginondb
import logging

print "Handelers", len(logging.getLogger('sqlalchemy').handlers)
print "Handelers2", len(logging.getLogger('sqlalchemy.engine').handlers)

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
        self.login={}
    
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
        if self.login.get('username', None) == username and \
           self.login.get('leginonDB', None) == leginonDB and \
           self.login.get('projectDB', None) == projectDB and \
           self.login.get('password', None) == password: return 
        #alternteUser = self.ui.alternateUserLineEdit.text()
        try:
            user = leginondb.projects_for_user(username, password, leginonDB, projectDB)#, alternteUser)
        except:
            _logger.exception("Error accessing project")
            self.ui.loginStackedWidget.setCurrentIndex(1)
        else:
            experiments = []#user.projects[0].experiments
            for i in xrange(len(user.projects)):
                experiments.extend(user.projects[i].experiments)
            header=['Session', 'Project', 'Images', 'Voltage', 'Pixel Size', 'Magnification', 'CS']
            data = []
            for exp in experiments:
                session = exp.session
                project=session.projects[0]
                if len(session.exposures) == 0: continue
                #if session.scope.instrument is None: continue
                voltage = session.scope.voltage/1000
                cs = session.scope.instrument.cs*1e3 if session.scope.instrument is not None else -1.0
                magnification = session.exposures[0].scope.magnification
                pixel_size = session.exposures[0].pixelsize*1e10
                data.append( (session.name, project.name, len(session.exposures), voltage, pixel_size, magnification, cs))
            self.ui.projectTableView.setModel(ListTableModel(data, header))
            self.ui.label.setText("Welcome "+str(user.fullname))
    
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
        #self.ui.alternateUserLineEdit.setText(settings.value('alternate-user'))
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
        #settings.setValue('alternate-user', self.ui.alternateUserLineEdit.text())
        settings.endGroup()

