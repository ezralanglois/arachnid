'''

.. todo:: Use background task rather than timer?


.. Created on Dec 4, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from pyui.LeginonUI import Ui_Form
from util.qt4_loader import QtGui,QtCore,qtSlot
from model.ListTableModel import ListTableModel
from ..metadata import leginondb
import multiprocessing
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class Widget(QtGui.QWidget): 
    ''' Display Controls for the LeginonUI
    '''
    
    def __init__(self, parent=None):
        "Initialize LeginonUI widget"
        
        QtGui.QWidget.__init__(self, parent)
        
        # Build window
        _logger.info("\rBuilding main window ...")
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.login={}
        
        self.ui.progressDialog = QtGui.QProgressDialog('Loading...', "", 0,5,self)
        self.ui.progressDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.progressDialog.findChildren(QtGui.QPushButton)[0].hide()
        #self.ui.progressDialog.setCancelButton(0) # might work
        #self.ui.progressDialog.setWindowFlags(self.ui.progressDialog.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint) #v5s
        self.ui.progressDialog.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowTitleHint | QtCore.Qt.CustomizeWindowHint)
        
        self.ui.loadTimer = QtCore.QTimer(self)
        self.ui.loadTimer.setInterval(500)
        self.ui.loadTimer.setSingleShot(False)
        self.ui.loadTimer.timeout.connect(self.on_loadTimer_timeout)
        self.header=None
        self.data=None
    
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
        
    def on_loadTimer_timeout(self):
        '''
        '''
        
        if self.qout is None: self.ui.loadTimer.stop()
        if self.data is None: self.data=[]
        while not self.qout.empty():
            try:
                val = self.qout.get(False)
            except multiprocessing.Queue.Empty:
                break
            if val is None:
                self.ui.loadTimer.stop()
                self.ui.projectTableView.setModel(ListTableModel(self.data, self.header))
                self.data=None
                self.header=None
                self.ui.progressDialog.hide()
                # check file locations
            elif not hasattr(val, '__iter__'):
                self.ui.progressDialog.setMaximum(val)
            else:
                self.data.append(val)
                self.ui.progressDialog.setValue(len(self.data))
    
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
        self.login['username']=username
        self.login['leginonDB']=leginonDB
        self.login['projectDB']=projectDB
        self.login['password']=password
        #alternteUser = self.ui.alternateUserLineEdit.text()
        try:
            user = leginondb.query_user_info(username, password, leginonDB, projectDB)#, alternteUser)
        except:
            _logger.exception("Error accessing project")
            self.ui.loginStackedWidget.setCurrentIndex(1)
        else:
            self.ui.progressDialog.show()
            self.ui.label.setText("Welcome "+str(user.fullname))
            self.header=['Session', 'Project', 'Images', 'Voltage', 'Pixel Size', 'Magnification', 'CS']
            self.data=[]
            self.qout = load_projects(user)
            self.ui.loadTimer.start()
            self.ui.progressDialog.setValue(0)
    
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
        self.ui.imagesPathLineEdit.setText(settings.value('imagespath'))
        self.ui.framesPathLineEdit.setText(settings.value('framespath'))
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
        settings.setValue('imagespath', self.ui.imagesPathLineEdit.text())
        settings.setValue('framespath', self.ui.framesPathLineEdit.text())
        #settings.setValue('alternate-user', self.ui.alternateUserLineEdit.text())
        settings.endGroup()

def load_projects(user):
    '''
    '''
    
    qout = multiprocessing.Queue()
    displayProcess = multiprocessing.Process(target=load_projects_worker, args=(user, qout))
    displayProcess.start()
    return qout

def load_projects_worker(user, qout):
    '''
    '''
    
    try:
        experiments = []#user.projects[0].experiments
        for i in xrange(len(user.projects)):
            experiments.extend(user.projects[i].experiments)
        #header=['Session', 'Project', 'Images', 'Voltage', 'Pixel Size', 'Magnification', 'CS']
        qout.put(len(experiments))
        for exp in experiments:
            session = exp.session
            project=session.projects[0]
            if len(session.exposures) == 0: continue
            voltage = session.scope.voltage/1000
            cs = session.scope.instrument.cs*1e3 if session.scope.instrument is not None else -1.0
            magnification = session.exposures[0].scope.magnification
            pixel_size = session.exposures[0].pixelsize*1e10
            '''
            print session.exposures[0].filename
            print session.exposures[0].mrcimage
            print session.exposures[0].norm_filename
            print session.exposures[0].norm_mrcimage
            print session.exposures[0].frame_list
            print
            print
            '''
            row = (session.name, project.name, len(session.exposures), voltage, pixel_size, magnification, cs)
            qout.put( row )
    finally:
        qout.put(None)
