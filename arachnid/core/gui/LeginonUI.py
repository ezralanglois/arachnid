'''

.. todo:: Use background task rather than timer?


.. Created on Dec 4, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from pyui.LeginonUI import Ui_Form
from util.qt4_loader import QtGui,QtCore,qtSlot, qtSignal, qtProperty
from model.ListTableModel import ListTableModel
from ..metadata import leginondb
import multiprocessing
import logging
import os
import traceback
import sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class Widget(QtGui.QWidget): 
    ''' Display Controls for the LeginonUI
    '''
    
    selectionChanged = qtSignal()
    loadFinished = qtSignal()
    
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
        
        self.ui.loadProjectTimer = QtCore.QTimer(self)
        self.ui.loadProjectTimer.setInterval(500)
        self.ui.loadProjectTimer.setSingleShot(False)
        self.ui.loadProjectTimer.timeout.connect(self.on_loadProjectTimer_timeout)
        
        self.ui.loadImageTimer = QtCore.QTimer(self)
        self.ui.loadImageTimer.setInterval(500)
        self.ui.loadImageTimer.setSingleShot(False)
        self.ui.loadImageTimer.timeout.connect(self.on_loadImageTimer_timeout)
        
        self.ui.reloadTableToolButton.triggered.connect(self.queryDatabaseForProjects)
        
        self.header=['Session', 'Project', 'Images', 'Voltage', 'Pixel Size', 'Magnification', 'CS', 'Image Name', 'Image Path', 'Frame Path']
        self.data=None
        self.images = []
    
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
        
        if index == 0: self.queryDatabaseForProjects()
        
    def on_loadProjectTimer_timeout(self):
        '''
        '''
        
        if self.qout is None: self.ui.loadProjectTimer.stop()
        if self.data is None: self.data=[]
        while not self.qout.empty():
            try:
                val = self.qout.get(False)
            except multiprocessing.Queue.Empty:
                break
            if val is None:
                self.ui.loadProjectTimer.stop()
                if self.ui.projectTableView.selectionModel() is not None:
                    self.ui.projectTableView.selectionModel().selectionChanged.disconnect(self.selectionChanged)
                self.ui.projectTableView.setModel(ListTableModel(self.data, self.header))
                self.ui.projectTableView.selectionModel().selectionChanged.connect(self.selectionChanged)
                self.data=None
                self.ui.progressDialog.hide()
                # check file locations
            elif not hasattr(val, '__iter__'):
                self.ui.progressDialog.setMaximum(val)
            else:
                self.data.append(val)
                self.ui.progressDialog.setValue(len(self.data))
        
    def on_loadImageTimer_timeout(self):
        '''
        '''
        
        if self.qout is None: self.ui.loadImageTimer.stop()
        if self.images is None: self.images=[]
        while not self.qout.empty():
            try:
                val = self.qout.get(False)
            except multiprocessing.Queue.Empty:
                break
            if val is None:
                self.ui.loadImageTimer.stop()
                self.loadFinished.emit()
                
            elif not hasattr(val, '__iter__'):
                self.ui.progressDialog.setMaximum(val)
            else:
                self.images.append(val)
                self.ui.progressDialog.setValue(len(self.images))
    
    def queryDatabaseForProjects(self):
        '''
        '''
        
        username = self.ui.usernameLineEdit.text()
        password = self.ui.passwordLineEdit.text()
        leginonDB = self.ui.leginonDBLineEdit.text()
        projectDB = self.ui.projectDBLineEdit.text()
        limit = self.ui.entryLimitSpinBox.value()
        if self.login.get('username', None) == username and \
           self.login.get('leginonDB', None) == leginonDB and \
           self.login.get('projectDB', None) == projectDB and \
           self.login.get('password', None) == password and \
           self.login.get('limit', None) == limit: return
        self.login['username']=username
        self.login['leginonDB']=leginonDB
        self.login['projectDB']=projectDB
        self.login['password']=password
        self.login['limit']=limit
        #alternteUser = self.ui.alternateUserLineEdit.text()
        try:
            user = leginondb.query_user_info(username, password, leginonDB, projectDB)#, alternteUser)
        except:
            _logger.exception("Error accessing project")
            exception_message(self, "Error accessing project")
            self.ui.loginStackedWidget.setCurrentIndex(1)
        else:
            self.ui.progressDialog.show()
            self.ui.label.setText("Welcome "+str(user.fullname))
            self.data=[]
            self.qout = load_projects(user, limit)
            self.ui.loadProjectTimer.start()
            self.ui.progressDialog.setValue(0)
    
    def queryDatabaseImages(self, session_name):
        '''
        '''
        
        username = self.ui.usernameLineEdit.text()
        password = self.ui.passwordLineEdit.text()
        leginonDB = self.ui.leginonDBLineEdit.text()
        projectDB = self.ui.projectDBLineEdit.text()
        try:
            session = leginondb.query_session_info(username, password, leginonDB, projectDB, session_name)#, alternteUser)
        except:
            _logger.exception("Error accessing images")
            exception_message(self, "Error accessing images")
        else:
            if session is None:
                _logger.error("Error accessing images")
                return
                
            self.ui.progressDialog.show()
            self.images=[]
            self.qout = load_images(session)
            self.ui.loadImageTimer.start()
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
            self.queryDatabaseForProjects()
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
    
    def registerPage(self, wizardPage):
        '''
        '''
        
        wizardPage.registerField("sessions*", self, "selected", QtCore.SIGNAL('selectionChanged()'))
    
    @qtProperty(int)
    def selected(self):
        '''
        '''
        
        if self.ui.projectTableView.selectionModel() is None or \
           len(self.ui.projectTableView.selectionModel().selectedIndexes()) == 0: 
            return -1
        return self.ui.projectTableView.selectionModel().selectedIndexes()[0].row()
    
    def validate(self):
        '''
        '''
        
        if len(self.images) == 0:
            index = self.ui.projectTableView.selectionModel().selectedIndexes()[0]
            data = self.ui.projectTableView.model().row(index)   
            self.queryDatabaseImages(data[0])
            return False
        return True
    
    def currentData(self):
        '''
        '''
        
        if self.ui.projectTableView.selectionModel() is None or \
           len(self.ui.projectTableView.selectionModel().selectedIndexes()) == 0: 
            return {}
        index = self.ui.projectTableView.selectionModel().selectedIndexes()[0]
        data = self.ui.projectTableView.model().row(index)        
        #self.header=['Session', 'Project', 'Images', 'Voltage', 'Pixel Size', 'Magnification', 'CS']
        vals = []
        for i, h in enumerate(self.header):
            vals.append((h, data[i]))
        
        images = ",".join([img[0] for img in self.images])
        vals.append(('Micrograph Files', images))
        if self.images[0][1] is not None:
            vals.append(('Gain Files', ",".join([img[1] for img in self.images])))
        return dict(vals)
        
    #def updateWizard(self):
    
def load_images(session):
    '''
    '''
    
    qout = multiprocessing.Queue()
    displayProcess = multiprocessing.Process(target=load_images_worker, args=(session, qout))
    displayProcess.start()
    return qout

def load_images_worker(session, qout):
    '''
    '''
    
    try:
        qout.put(len(session.exposures))
        frame_path = session.frame_path
        image_path = session.image_path
        image_ext = '.mrc'
        if frame_path is None:
            frame_ext = image_ext
            frame_path = image_path
        else:
            frame_ext = '.frames.mrc'
        if frame_ext == image_ext:
            for image in session.exposures:
                row = (os.path.join(frame_path, image.filename+frame_ext), None)
                qout.put( row )
        else:
            for image in session.exposures:
                row = (os.path.join(frame_path, image.filename+frame_ext), os.path.join(image_path, image.norm_filename+image_ext))
                qout.put( row )
    finally:
        qout.put(None)
        
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
    
def exception_message(parent, msg):
    '''
    '''
    
    exc_type, exc_value = sys.exc_info()[:2]
    error_message(parent, msg, traceback.format_exception_only(exc_type, exc_value)[0])
        
def load_projects(user, limit):
    '''
    '''
    
    qout = multiprocessing.Queue()
    displayProcess = multiprocessing.Process(target=load_projects_worker, args=(user, limit, qout))
    displayProcess.start()
    return qout

def load_projects_worker(user, limit, qout):
    '''
    '''
    
    try:
        experiments = []#user.projects[0].experiments
        for i in xrange(len(user.projects)):
            experiments.extend(user.projects[i].experiments)
        #header=['Session', 'Project', 'Images', 'Voltage', 'Pixel Size', 'Magnification', 'CS']
        if len(experiments) > limit: experiments = experiments[:limit]
        qout.put(len(experiments))
        for exp in experiments:
            session = exp.session
            project=session.projects[0]
            if len(session.exposures) == 0: continue
            voltage = session.scope.voltage/1000
            cs = session.scope.instrument.cs*1e3 if session.scope.instrument is not None else -1.0
            magnification = session.exposures[0].scope.magnification
            pixel_size = session.exposures[0].pixelsize*1e10
            filename = session.exposures[0].mrcimage
            frame_path = session.frame_path
            image_path = session.image_path
            #type = session.exposures[0].camera
            
            '''
            print session.exposures[0].filename
            print session.exposures[0].mrcimage
            print session.exposures[0].norm_filename
            print session.exposures[0].norm_mrcimage
            print session.exposures[0].frame_list
            print
            print
            '''
            row = (session.name, project.name, len(session.exposures), voltage, pixel_size, magnification, cs, filename, image_path, frame_path)
            qout.put( row )
    finally:
        qout.put(None)
