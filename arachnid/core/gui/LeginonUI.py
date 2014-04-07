'''


.. Created on Dec 4, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from pyui.LeginonUI import Ui_Form
from util.qt4_loader import QtGui,QtCore,qtSlot, qtSignal, qtProperty
from model.ListTableModel import ListTableModel
from util import BackgroundTask
from util import messagebox
from ..metadata import leginondb
import base64
import logging
import os
import getpass
import hashlib

#import functools

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
    captureScreen = qtSignal(int)
    
    def __init__(self, parent=None, helpDialog=None):
        "Initialize LeginonUI widget"
        
        QtGui.QWidget.__init__(self, parent)
        
        # Build window
        _logger.info("\rBuilding main window ...")
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.login={}
        self.helpDialog=helpDialog
        self.startTabIndex=0
        
        self.ui.progressDialog = QtGui.QProgressDialog('Loading...', "", 0,5,self)
        self.ui.progressDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.ui.progressDialog.findChildren(QtGui.QPushButton)[0].hide()
        #self.ui.progressDialog.setCancelButton(0) # might work
        #self.ui.progressDialog.setWindowFlags(self.ui.progressDialog.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint) #v5s
        self.ui.progressDialog.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowTitleHint | QtCore.Qt.CustomizeWindowHint)
        
        self.ui.reloadTableToolButton.clicked.connect(self.queryDatabaseForProjects)
        self.taskUpdated.connect(self.updateProgress)
        
        self.ui.leginonHostnameLineEdit.editingFinished.connect(lambda: self.ui.projectHostnameLineEdit.setText(self.ui.leginonHostnameLineEdit.text()) if self.ui.projectHostnameLineEdit.text() == "" else None)
        
        header = [('Session', None), ('Project', None), ('Images', None), 
                  ('Voltage', 'voltage'), ('Pixel Size', 'apix'), 
                  ('Magnification', None), ('CS', 'cs'), ('Image Name', None), 
                  ('Image Path', None), ('Frame Path', None)]
        self.header=[h[0] for h in header]
        self.headermap = dict([h for h in header if h[1] is not None])
        self.images = []
        
        self.ui.projectTableView.setModel(ListTableModel([], self.header, None, self))
        selmodel=self.ui.projectTableView.selectionModel()
        selmodel.selectionChanged.connect(self.selectionChanged)
        
    def initializePage(self):
        '''
        '''
        
        self.ui.loginStackedWidget.setCurrentIndex(self.startTabIndex)
    
    @qtSlot()
    def on_userInformationToolButton_clicked(self):
        '''
        '''
        
        if self.helpDialog is not None:
            self.helpDialog.setHTML(self.ui.changeUserPushButton.toolTip())
            self.helpDialog.show()
        else:
            QtGui.QToolTip.showText(self.ui.changeUserPushButton.mapToGlobal(QtCore.QPoint(0,0)), self.ui.changeUserPushButton.toolTip())
    
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
        
        fields = [self.ui.usernameLineEdit, self.ui.dbPasswordLineEdit, self.ui.dbUsernameLineEdit, self.ui.projectHostnameLineEdit, 
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
        
        self.captureScreen.emit(index+1)
        if index == 0: self.queryDatabaseForProjects()
    
    def queryDatabaseForProjects(self, dummy=None):
        '''
        '''
        
        targetuser = self.ui.usernameLineEdit.text()
        targetpass = hashlib.md5(self.ui.passwordLineEdit.text()).hexdigest()
        username = self.ui.dbUsernameLineEdit.text()
        password = self.ui.dbPasswordLineEdit.text()
        prjhost = self.ui.projectHostnameLineEdit.text() if self.ui.projectHostnameLineEdit.text() != "" else self.ui.leginonHostnameLineEdit.text()
        leginonDB = self.ui.leginonHostnameLineEdit.text()+'/'+self.ui.leginonDBNameLineEdit.text()
        projectDB = prjhost+'/'+self.ui.projectDBNameLineEdit.text()
        limit = self.ui.entryLimitSpinBox.value()
        if self.login.get('username', None) == username and \
           self.login.get('leginonDB', None) == leginonDB and \
           self.login.get('projectDB', None) == projectDB and \
           self.login.get('password', None) == password and \
           self.login.get('targetpass', None) == targetpass and \
           self.login.get('targetuser', None) == targetuser and \
           self.login.get('limit', None) == limit: return
        #alternteUser = self.ui.alternateUserLineEdit.text()
        try:
            user, experiments = leginondb.query_user_info(username, password, leginonDB, projectDB, targetuser, targetpass)#, alternteUser)
        except leginondb.AuthenticationError:
            messagebox.error_message(self, "Username or password incorrect!")
            #self.ui.loginStackedWidget.setCurrentIndex(1)
            self.startTabIndex=1
        except:
            _logger.exception("Error accessing project")
            messagebox.exception_message(self, "Error accessing project")
            #self.ui.loginStackedWidget.setCurrentIndex(1)
            self.startTabIndex=1
        else:
            self.login['username']=username
            self.login['leginonDB']=leginonDB
            self.login['projectDB']=projectDB
            self.login['password']=password
            self.login['limit']=limit
            self.login['targetuser']=targetuser
            self.login['targetpass']=targetpass
            self.ui.progressDialog.show()
            self.ui.label.setText("Welcome "+str(user.fullname))
            self.taskFinished.connect(self.projectFinished)
            self.taskError.connect(self.projectLoadError)
            experiment_list = []
            cnt = 0
            for i in xrange(len(experiments)):
                if len(experiments[i].session.exposures) == 0: continue
                experiment_list.append(experiments[i])
                cnt += 1
                if cnt >= limit: break
            self.task = BackgroundTask.launch_mp(self, load_projects_iter, experiment_list)
    
    def projectFinished(self, sessions):
        '''
        '''
        
        self.ui.projectTableView.model().setData(sessions)        
        self.ui.progressDialog.hide()
        self.taskFinished.disconnect(self.projectFinished)
        self.taskError.disconnect(self.projectLoadError)
        self.task = None
    
    def projectLoadError(self, exception):
        '''
        '''
        
        self.ui.progressDialog.hide()
        messagebox.exception_message(self, "Error accessing projects", exception)
        self.taskFinished.disconnect(self.projectFinished)
        self.taskError.disconnect(self.projectLoadError)
        self.task = None
    
    def updateProgress(self, val):
        
        if hasattr(val, '__iter__'):
            if len(val) == 1 and not hasattr(val, '__iter__'):
                self.ui.progressDialog.setMaximum(val[0])
        else:
            self.ui.progressDialog.setValue(val)
    
    def queryDatabaseImages(self, session_name):
        '''
        '''
        
        username = self.ui.dbUsernameLineEdit.text()
        password = self.ui.dbPasswordLineEdit.text()
        #targetuser = self.ui.usernameLineEdit.text()
        #targetpass = self.ui.passwordLineEdit.text()
        prjhost = self.ui.projectHostnameLineEdit.text() if self.ui.projectHostnameLineEdit.text() != "" else self.ui.leginonHostnameLineEdit.text()
        leginonDB = self.ui.leginonHostnameLineEdit.text()+'/'+self.ui.leginonDBNameLineEdit.text()
        projectDB = prjhost+'/'+self.ui.projectDBNameLineEdit.text()
        try:
            session = leginondb.query_session_info(username, password, leginonDB, projectDB, session_name)#, alternteUser)
        except:
            _logger.exception("Error accessing images")
            messagebox.exception_message(self, "Error accessing images")
            self.ui.loginStackedWidget.setCurrentIndex(1)
        else:
            if len(session)==0:
                _logger.error("Error accessing images")
                return
                
            self.taskFinished.connect(self.imageLoadFinished)
            self.taskError.connect(self.imageLoadError)
            self.task = BackgroundTask.launch_mp(self, load_images_iter, session)
            self.ui.progressDialog.show()
            self.ui.progressDialog.setValue(0)
    
    def imageLoadError(self, exception):
        '''
        '''
        
        messagebox.exception_message(self, "Error accessing images", exception)
        self.ui.progressDialog.hide()
        self.taskFinished.disconnect(self.imageLoadFinished)
        self.taskError.disconnect(self.imageLoadError)
        self.task = None
    
    def imageLoadFinished(self, images):
        '''
        '''
        
        self.images = images
        self.ui.progressDialog.hide()
        self.taskFinished.disconnect(self.imageLoadFinished)
        self.taskError.disconnect(self.imageLoadError)
        self.loadFinished.emit()
        self.task = None
    
    def readSettings(self):
        '''
        '''
        
        settings = QtCore.QSettings()
        settings.beginGroup("LeginonUI")
        val = settings.value('leginonDB')
        if val: self.ui.leginonHostnameLineEdit.setText(val)
        val = settings.value('leginonPath')
        if val: self.ui.leginonDBNameLineEdit.setText(val)
        val = settings.value('projectDB')
        if val: self.ui.projectHostnameLineEdit.setText(val)
        val = settings.value('projectPath')
        if val: self.ui.projectDBNameLineEdit.setText(val)
        self.ui.usernameLineEdit.setText(settings.value('targetuser'))
        self.ui.passwordLineEdit.setText(settings.value('targetpass'))
        
        val = settings.value('username')
        if val: self.ui.dbUsernameLineEdit.setText(val)
        val = settings.value('password')
        if val: self.ui.dbPasswordLineEdit.setText(base64.b64decode(val))
        
        #self.ui.alternateUserLineEdit.setText(settings.value('alternate-user'))
        settings.endGroup()
        if self.ui.usernameLineEdit.text() == "":
            self.ui.usernameLineEdit.setText(getpass.getuser())
        
    def writeSettings(self):
        '''
        '''
        
        settings = QtCore.QSettings()
        settings.beginGroup("LeginonUI")
        settings.setValue('leginonDB', self.ui.leginonHostnameLineEdit.text())
        settings.setValue('leginonPath', self.ui.leginonDBNameLineEdit.text())
        settings.setValue('projectDB', self.ui.projectHostnameLineEdit.text())
        settings.setValue('projectPath', self.ui.projectDBNameLineEdit.text())
        settings.setValue('targetuser', self.ui.usernameLineEdit.text())
        settings.setValue('targetpass', self.ui.passwordLineEdit.text())
        settings.setValue('username', self.ui.dbUsernameLineEdit.text())
        settings.setValue('password', base64.b64encode(self.ui.dbPasswordLineEdit.text()))
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
        
        model = self.ui.projectTableView.selectionModel()
        #index = model.selectedIndexes()[0]
        selectedRows = [index for index in model.selectedIndexes() if index.column() == 0]
        data = [self.ui.projectTableView.model().row(index)[0] for index in selectedRows]
        self.queryDatabaseImages(data)#[0])
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
        
        
        if len(self.images) > 1:
            choices=[]
            keys = self.images.keys()
            keys = [k[1] for k in sorted(zip([self.images[key] for key in keys], keys))]
            for key in keys:
                choices.append("%f (%d)"%(key, len(self.images[key])))
            apix = QtGui.QInputDialog.getItem(self, "Multiple exposure sizes found!", "Pixel sizes (# exposures):", choices)
            if isinstance(apix, tuple): apix=apix[0]
            if apix is None: return None
            apix = float(apix[:apix.find('(')].strip())
            image_list = self.images[apix]
            for i in xrange(len(vals)):
                if vals[i][0] == 'apix': vals[i]=('apix', apix)
        else: image_list=self.images[self.images.keys()[0]]
        
        images = ",".join([img[0] for img in image_list])
        vals.append(('input_files', images))
        if image_list[0][1] is not None:
            vals.append(('gain_files', ",".join([img[1] for img in image_list])))
        return dict(vals)
        
    #def updateWizard(self):

def load_images_iter(sessions):
    '''
    '''
    total = 0
    for session in sessions: total += len(session.exposures)
    yield total
    
    images = {}
    total=0
    for session in sessions:
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
                apix = image.pixelsize*1e10
                if apix not in images: images[apix]=[]
                images[apix].append(row)
                total += 1
                yield total
        else:
            norm_file="None"
            norm_id="None"
            for image in session.exposures:
                if image.norm_id is None: 
                    _logger.warning("Skipping %s - no gain normalization listed in database"%image.filename)
                    continue
                if image.norm_id != norm_id:
                    norm_path=image.norm_path
                    norm_file = os.path.join(norm_path, image.norm_filename+image_ext)
                    norm_id=image.norm_id
                exposure=os.path.join(frame_path, image.filename+frame_ext)
                if not os.path.exists(exposure) and os.path.exists(exposure+'.bz2'): exposure += '.bz2'
                row = (exposure, norm_file)
                    
                apix = image.pixelsize*1e10
                if apix not in images: images[apix]=[]
                images[apix].append(row)
                total += 1
                yield total
    yield images


def load_projects_iter(experiments):
    yield (len(experiments), )
    rows=[]
    for exp in experiments:
        session = exp.session
        project=session.projects[0]
        if len(session.exposures) == 0: continue
        voltage = session.scopes[0].voltage/1000
        cs = session.scopes[0].instrument.cs*1e3 if session.scopes[0].instrument is not None else -1.0
        magnification = session.exposures[0].scope.magnification
        pixel_size = session.exposures[0].pixelsize*1e10
        filename = str(session.exposures[0].mrcimage)
        frame_path = session.frame_path
        image_path = session.image_path
        #type = session.exposures[0].camera
        if frame_path is not None: frame_path=str(frame_path)
        rows.append((str(session.name), str(project.name), len(session.exposures), float(voltage), float(pixel_size), float(magnification), float(cs), str(filename), str(image_path), frame_path))
        yield len(rows)
    yield rows

