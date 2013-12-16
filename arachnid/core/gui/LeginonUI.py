'''

.. todo:: Combine sessions?

.. todo:: Check mag?


.. Created on Dec 4, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from pyui.LeginonUI import Ui_Form
from util.qt4_loader import QtGui,QtCore,qtSlot, qtSignal, qtProperty
from model.ListTableModel import ListTableModel
from util import BackgroundTask
from ..metadata import leginondb
import base64
import logging
import os
import traceback
import sys
import getpass

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class Widget(QtGui.QWidget): 
    ''' Display Controls for the LeginonUI
    '''
    
    selectionChanged = qtSignal()
    loadFinished = qtSignal()
    taskFinished = qtSignal(object)
    taskUpdated = qtSignal(object)
    taskError = qtSignal(object)
    
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
        
        self.ui.reloadTableToolButton.triggered.connect(self.queryDatabaseForProjects)
        self.taskUpdated.connect(self.updateProgress)
        
        self.ui.leginonHostnameLineEdit.editingFinished.connect(lambda: self.ui.projectHostnameLineEdit.setText(self.ui.leginonHostnameLineEdit.text()) if self.ui.projectHostnameLineEdit.text() == "" else None)
        
        header = [('Session', None), ('Project', None), ('Images', None), 
                  ('Voltage', 'voltage'), ('Pixel Size', 'apix'), 
                  ('Magnification', None), ('CS', 'cs'), ('Image Name', None), 
                  ('Image Path', None), ('Frame Path', None)]
        self.header=[h[0] for h in header]
        self.headermap = dict([h for h in header if h[1] is not None])
        self.images = []
        
        self.ui.projectTableView.setModel(ListTableModel([], self.header, self))
        selmodel=self.ui.projectTableView.selectionModel()
        selmodel.selectionChanged.connect(self.selectionChanged)
    
    def showEvent(self, evt):
        '''Window close event triggered - save project and global settings 
        
        :Parameters:
            
        evt : QCloseEvent
              Event for to close the main window
        '''
        
        # Load the settings
        _logger.info("\rLoading settings ...")
        self.readSettings()
        self.on_loginPushButton_clicked()
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
    
    @qtSlot()
    def on_changeUserPushButton_clicked(self):
        '''
        '''
        
        self.ui.loginStackedWidget.setCurrentIndex(1)
    
    @qtSlot()
    def on_loginPushButton_clicked(self):
        '''
        '''
        
        fields = [self.ui.usernameLineEdit, self.ui.passwordLineEdit, self.ui.projectHostnameLineEdit, 
                  self.ui.leginonHostnameLineEdit, self.ui.leginonDBNameLineEdit,
                  self.ui.projectDBNameLineEdit]
        for field in fields:
            if field.text() == "":
                field.setStyleSheet("border: 2px solid red")
                return
        
        self.writeSettings()
        self.ui.loginStackedWidget.setCurrentIndex(0)
    
    @qtSlot(int)
    def on_loginStackedWidget_currentChanged(self, index):
        '''
        '''
        
        if index == 0: self.queryDatabaseForProjects()
    
    def queryDatabaseForProjects(self):
        '''
        '''
        #
        
        username = self.ui.usernameLineEdit.text()
        password = self.ui.passwordLineEdit.text()
        prjhost = self.ui.projectHostnameLineEdit.text() if self.ui.projectHostnameLineEdit.text() != "" else self.ui.leginonHostnameLineEdit.text()
        leginonDB = self.ui.leginonHostnameLineEdit.text()+'/'+self.ui.leginonDBNameLineEdit.text()
        projectDB = prjhost+'/'+self.ui.projectDBNameLineEdit.text()
        limit = self.ui.entryLimitSpinBox.value()
        if self.login.get('username', None) == username and \
           self.login.get('leginonDB', None) == leginonDB and \
           self.login.get('projectDB', None) == projectDB and \
           self.login.get('password', None) == password and \
           self.login.get('limit', None) == limit: return
        #alternteUser = self.ui.alternateUserLineEdit.text()
        try:
            user, experiments = leginondb.query_user_info(username, password, leginonDB, projectDB)#, alternteUser)
        except:
            _logger.exception("Error accessing project")
            exception_message(self, "Error accessing project")
            self.ui.loginStackedWidget.setCurrentIndex(1)
        else:
            self.login['username']=username
            self.login['leginonDB']=leginonDB
            self.login['projectDB']=projectDB
            self.login['password']=password
            self.login['limit']=limit
            self.ui.progressDialog.show()
            self.ui.label.setText("Welcome "+str(user.fullname))
            self.taskFinished.connect(self.projectFinished)
            self.taskError.connect(self.projectLoadError)
            BackgroundTask.launch_mp(self, load_projects_iter, experiments[:limit])#, limit)
    
    def projectFinished(self, sessions):
        '''
        '''
        
        self.ui.projectTableView.model().setData(sessions)        
        self.ui.progressDialog.hide()
        self.taskFinished.disconnect(self.projectFinished)
        self.taskError.disconnect(self.projectLoadError)
    
    def projectLoadError(self, exception):
        '''
        '''
        
        self.ui.progressDialog.hide()
        exception_message(self, "Error accessing projects", exception)
        self.taskFinished.disconnect(self.projectFinished)
        self.taskError.disconnect(self.projectLoadError)
    
    def updateProgress(self, val):
        
        if hasattr(val, '__iter__'):
            if len(val) == 1 and not hasattr(val, '__iter__'):
                self.ui.progressDialog.setMaximum(val[0])
        else:
            self.ui.progressDialog.setValue(val)
    
    def queryDatabaseImages(self, session_name):
        '''
        '''
        
        username = self.ui.usernameLineEdit.text()
        password = self.ui.passwordLineEdit.text()
        prjhost = self.ui.projectHostnameLineEdit.text() if self.ui.projectHostnameLineEdit.text() != "" else self.ui.leginonHostnameLineEdit.text()
        leginonDB = self.ui.leginonHostnameLineEdit.text()+'/'+self.ui.leginonDBNameLineEdit.text()
        projectDB = prjhost+'/'+self.ui.projectDBNameLineEdit.text()
        try:
            session = leginondb.query_session_info(username, password, leginonDB, projectDB, session_name)#, alternteUser)
        except:
            _logger.exception("Error accessing images")
            exception_message(self, "Error accessing images")
            self.ui.loginStackedWidget.setCurrentIndex(1)
        else:
            if session is None:
                _logger.error("Error accessing images")
                return
                
            self.taskFinished.connect(self.imageLoadFinished)
            self.taskError.connect(self.imageLoadError)
            BackgroundTask.launch_mp(self, load_images_iter, session)
            self.ui.progressDialog.show()
            self.ui.progressDialog.setValue(0)
    
    
    def imageLoadError(self, exception):
        '''
        '''
        
        exception_message(self, "Error accessing images", exception)
        self.ui.progressDialog.hide()
        self.taskFinished.disconnect(self.imageLoadFinished)
        self.taskError.disconnect(self.imageLoadError)
    
    def imageLoadFinished(self, images):
        '''
        '''
        
        self.images = images
        self.ui.progressDialog.hide()
        self.taskFinished.disconnect(self.imageLoadFinished)
        self.taskError.disconnect(self.imageLoadError)
        self.loadFinished.emit()
    
    def readSettings(self):
        '''
        '''
        
        print 'read settings'
        settings = QtCore.QSettings()
        settings.beginGroup("LeginonUI")
        self.ui.leginonHostnameLineEdit.setText(settings.value('leginonDB'))
        self.ui.leginonDBNameLineEdit.setText(settings.value('leginonPath'))
        self.ui.projectHostnameLineEdit.setText(settings.value('projectDB'))
        self.ui.projectDBNameLineEdit.setText(settings.value('projectPath'))
        self.ui.usernameLineEdit.setText(settings.value('username'))
        self.ui.passwordLineEdit.setText(base64.b64decode(settings.value('password')))
        #self.ui.alternateUserLineEdit.setText(settings.value('alternate-user'))
        settings.endGroup()
        if self.ui.usernameLineEdit.text() == "":
            self.ui.usernameLineEdit.setText(getpass.getuser())
        
    def writeSettings(self):
        '''
        '''
        
        print 'write settings - leginon'
        settings = QtCore.QSettings()
        settings.beginGroup("LeginonUI")
        settings.setValue('leginonDB', self.ui.leginonHostnameLineEdit.text())
        settings.setValue('leginonPath', self.ui.leginonDBNameLineEdit.text())
        settings.setValue('projectDB', self.ui.projectHostnameLineEdit.text())
        settings.setValue('projectPath', self.ui.projectDBNameLineEdit.text())
        settings.setValue('username', self.ui.usernameLineEdit.text())
        settings.setValue('password', base64.b64encode(self.ui.passwordLineEdit.text()))
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
        
        model = self.ui.projectTableView.selectionModel()
        if model is None or \
           len(model.selectedIndexes()) == 0: 
            return -1
        return model.selectedIndexes()[0].row()
    
    def validate(self):
        '''
        '''
        
        if len(self.images) == 0:
            model = self.ui.projectTableView.selectionModel()
            index = model.selectedIndexes()[0]
            data = self.ui.projectTableView.model().row(index)   
            self.queryDatabaseImages(data[0])
            return False
        return True
    
    def currentData(self):
        '''
        '''
        
        model = self.ui.projectTableView.selectionModel()
        if model is None or \
           len(model.selectedIndexes()) == 0: 
            return {}
        indexes = model.selectedIndexes()
        index = indexes[0]
        dmodel = self.ui.projectTableView.model()
        data = dmodel.row(index)        
        vals = []
        for key, val in self.headermap.iteritems():
            vals.append((val, data[self.header.index(key)]))
        images = ",".join([img[0] for img in self.images])
        vals.append(('input_files', images))
        if self.images[0][1] is not None:
            vals.append(('gain_files', ",".join([img[1] for img in self.images])))
        return dict(vals)
        
    #def updateWizard(self):
        
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

def load_images_iter(session):
    '''
    '''
    
    yield (len(session.exposures), )
    images = []
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
            images.append(row)
            yield len(images)
    else:
        for image in session.exposures:
            row = (os.path.join(frame_path, image.filename+frame_ext), os.path.join(image_path, image.norm_filename+image_ext))
            images.append(row)
            yield len(images)
    yield images

'''
def load_projects_iter(user, limit):
    
    _logger.exception("load_projects_iter-1")
    experiments = []#user.projects[0].experiments
    projects = user.projects
    for i in xrange(len(projects)):
        experiments.extend(projects[i].experiments)
    #header=['Session', 'Project', 'Images', 'Voltage', 'Pixel Size', 'Magnification', 'CS']
    if len(experiments) > limit: experiments = experiments[:limit]
'''
def load_projects_iter(experiments):
    yield (len(experiments), )
    rows=[]
    for exp in experiments:
        session = exp.session
        project=session.projects[0]
        if len(session.exposures) == 0: continue
        voltage = session.scope.voltage/1000
        cs = session.scope.instrument.cs*1e3 if session.scope.instrument is not None else -1.0
        magnification = session.exposures[0].scope.magnification
        pixel_size = session.exposures[0].pixelsize*1e10
        filename = str(session.exposures[0].mrcimage)
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
        if frame_path is not None: frame_path=str(frame_path)
        rows.append((str(session.name), str(project.name), len(session.exposures), float(voltage), float(pixel_size), float(magnification), float(cs), str(filename), str(image_path), frame_path))
        yield len(rows)
    yield rows

