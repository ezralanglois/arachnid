'''

SessionData - name
SessionData.name
SessionData.DEF_id

tquery = leginondata.AcquisitionImageTargetData(list=targetlistdata)
imquery = leginondata.AcquisitionImageData(target=tquery)


Gain Reference
--------------

cam = leginondata.InstrumentData(name='GatanK2Counting').query(results=1)[0]
camem = leginondata.CameraEMData(ccdcamera=cam)

refpath = None
norm = leginondata.NormImageData(camera=camem).query(results=1)
if norm:
  # If norm image is indeed saved as a database entry, then database would know where it is saved
  refpath = norm[0]['session']['image path']
  print 'norm image path', refpath
  print 'file can be found', os.path.isfile(os.path.join(refpath,norm[0]['filename']+'.mrc'))

Workflow
--------

- Select Micrograph Data - Image Display
- Select reference optional
- Select gain reference - if movie mode data
- Microscope parameters (add valid defocus range)
- Individual settings
- Run monitor

6.206

.. Created on Sep 5, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from LeginonUI import Widget as LeginonUI
from ReferenceUI import Widget as ReferenceUI
from Monitor import Widget as MonitorUI
from SettingsEditor import TabWidget as SettingsUI

from pyui.ProjectUI import Ui_ProjectWizard
from util.qt4_loader import QtGui, qtSlot, QtCore, qtProperty, qtSignal
from arachnid.util import project
from ..image import ndimage_file
from ..parallel import openmp
import logging
import multiprocessing
import arachnid
import os
import glob
import sys
sys.setrecursionlimit(10000)


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class MainWindow(QtGui.QWizard):
    '''
    '''
    
    micrographFilesUpdated = qtSignal()
    gainFilesUpdated = qtSignal()
    
    def __init__(self, parent=None):
        '''
        '''
        
        
        QtGui.QWizard.__init__(self, parent)
        
        self.ui = Ui_ProjectWizard()
        self.ui.setupUi(self)
        #self.subPages={}
        self.lastpath = str(QtCore.QDir.currentPath())
        self.micrograph_files = []
        self.gain_files = []
        self.parameters=[]
        
        
        version = arachnid.__version__
        n=version.find('_')
        if n != -1: version = version[:n]
        self.setWindowTitle("Arachnid - Workflow Creation Wizard - v%s"%version)

        
        self.setPixmap(QtGui.QWizard.WatermarkPixmap, QtGui.QPixmap(':/icons/icons/icon256x256.png'))
        self.setPixmap(QtGui.QWizard.BackgroundPixmap, QtGui.QPixmap(':/icons/icons/icon256x256.png'))
        
        ########################################################################################################################################
        ###### Introduction Page
        ########################################################################################################################################
        self.ui.introductionPage.setTitle('Welcome to Arachnid - v%s'%version)
        
        ########################################################################################################################################
        ###### Question Page
        ########################################################################################################################################
        self.currentIdChanged.connect(self.onCurrentIDChanged)
        self.ui.yesLeginonPushButton.clicked.connect(self.next)
        self.ui.noLeginonPushButton.clicked.connect(self.next)
        self.ui.yesReferencePushButton.clicked.connect(self.next)
        self.ui.noReferencePushButton.clicked.connect(self.next)
        
        ########################################################################################################################################
        ###### Leginon Page
        ########################################################################################################################################
        self.ui.leginonWidget = LeginonUI(self)
        self.ui.leginonDBLayout.addWidget(self.ui.leginonWidget)
        #self.subPages[self.idOf(self.ui.leginonDBPage)]=self.ui.leginonWidget
        self.ui.leginonWidget.registerPage(self.ui.leginonDBPage)
        self.ui.leginonWidget.loadFinished.connect(self.next)
        
        ########################################################################################################################################
        ###### Reference Page
        ########################################################################################################################################
        self.ui.referenceWidget = ReferenceUI(self)
        self.ui.referenceLayout.addWidget(self.ui.referenceWidget)
        #self.subPages[self.idOf(self.ui.referencePage)]=self.ui.referenceWidget    
        self.ui.referenceWidget.registerPage(self.ui.referencePage) 
        #self.ui.referencePage.registerField("referenceEdit*", self.ui.referenceWidget.ui.referenceLineEdit)
        
        ########################################################################################################################################
        ###### Monitor Page
        ########################################################################################################################################
        self.ui.monitorWidget = MonitorUI(self)
        self.ui.monitorLayout.addWidget(self.ui.monitorWidget)
        
        ########################################################################################################################################
        ###### Fine Settings Page
        ########################################################################################################################################
        self.ui.settingsTabWidget = SettingsUI(self)
        self.ui.settingsHorizontalLayout.addWidget(self.ui.settingsTabWidget)
        self.ui.workflowListView.setModel(self.ui.monitorWidget.model())
        selmodel = self.ui.workflowListView.selectionModel()
        selmodel.currentChanged.connect(self.ui.settingsTabWidget.settingsChanged)
        
        
        self.job_status_icons=[QtGui.QIcon(f) for f in [':/mini/mini/clock.png', ':/mini/mini/arrow_refresh.png', ':/mini/mini/tick.png', ':/mini/mini/cross.png']]
        '''
        for filename in [':/mini/mini/clock.png', ':/mini/mini/arrow_refresh.png', ':/mini/mini/tick.png', ':/mini/mini/cross.png']:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(filename), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.jobStatusIcons.append(icon)
        '''
        
        ########################################################################################################################################
        ###### Manual Settings Page
        ########################################################################################################################################
        self.ui.manualSettingsPage.registerField(self.param("apix*"), self.ui.pixelSizeDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.manualSettingsPage.registerField(self.param("voltage*"), self.ui.voltageDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.manualSettingsPage.registerField(self.param("cs*"), self.ui.csDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.manualSettingsPage.registerField(self.param("input_files*"), self, 'micrographFiles', QtCore.SIGNAL('micrographFilesUpdated()'))
        self.ui.manualSettingsPage.registerField(self.param("gain_files"), self, 'gainFiles', QtCore.SIGNAL('gainFilesUpdated()'))
        self.ui.manualSettingsPage.registerField(self.param("gain_file"), self, 'gainFile')
        self.ui.manualSettingsPage.registerField(self.param("invert"), self.ui.invertCheckBox)
        
        ########################################################################################################################################
        ###### Additional Settings Page
        ########################################################################################################################################
        self.ui.particleSizeDoubleSpinBox.valueChanged.connect(lambda x: self.ui.particleDiameterPixelLabel.setText("%d (Px)"%(int(x/self.ui.pixelSizeDoubleSpinBox.value()))) if self.ui.pixelSizeDoubleSpinBox.value() > 0 else None)
        self.ui.windowAngstromDoubleSpinBox.valueChanged.connect(lambda x: self.ui.windowWidthAnstromsLabel.setText("%d (Px)"%(x/self.ui.pixelSizeDoubleSpinBox.value())) if self.ui.pixelSizeDoubleSpinBox.value() > 0 else None)
        self.ui.maskDiameterDoubleSpinBox.valueChanged.connect(lambda x: self.ui.maskDiameterPixelLabel.setText("%d (Px)"%(int(x/self.ui.pixelSizeDoubleSpinBox.value()))) if self.ui.pixelSizeDoubleSpinBox.value() > 0 else None)
        self.ui.particleSizeDoubleSpinBox.valueChanged.connect(lambda x: self.ui.windowAngstromDoubleSpinBox.setValue(x*1.4))
        self.ui.particleSizeDoubleSpinBox.valueChanged.connect(lambda x: self.ui.maskDiameterDoubleSpinBox.setValue(x*1.2))
        self.ui.additionalSettingsPage.registerField(self.param("particle_diameter*"), self.ui.particleSizeDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.additionalSettingsPage.registerField(self.param("window_actual*"), self.ui.windowAngstromDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.additionalSettingsPage.registerField(self.param("mask_diameter*"), self.ui.maskDiameterDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.additionalSettingsPage.registerField(self.param('worker_count'), self.ui.workerCountSpinBox)
        self.ui.additionalSettingsPage.registerField(self.param('thread_count'), self.ui.threadCountSpinBox)
        self.ui.additionalSettingsPage.registerField(self.param("window"), self, "window")
        
        thread_count = 1
        if openmp.is_openmp_enabled():
            thread_count = openmp.get_num_procs()
        else:
            try: thread_count=multiprocessing.cpu_count()
            except: pass
        self.ui.workerCountSpinBox.setValue(thread_count)
    
    def param(self, name):
        '''
        '''
        if name[-1]=='*':
            self.parameters.append(name[:len(name)-1])
        else:
            self.parameters.append(name)
        
        return name

    ########################################################################################################################################
    ###### Micrograph filename controls
    ########################################################################################################################################
    def micrographFiles(self):
        '''
        '''
        
        return ",".join(self.micrograph_files)
    
    def setMicrographFiles(self, files):
        '''
        '''
        
        _logger.error("setMicrographFiles-here1")
        try:"+"+files
        except: pass
        else: files = files.split(',')
        self.ui.micrographComboBox.blockSignals(True)
        self.ui.micrographComboBox.clear()
        self.ui.micrographComboBox.addItems(files)
        self._updateMicrographFiles(files)
        
        if len(files) > 0:
            self.ui.micrographComboBox.lineEdit().setText(self.ui.micrographComboBox.itemText(0))
        else:
            self.ui.micrographComboBox.lineEdit().setText("")
        self.ui.micrographComboBox.blockSignals(False)
    
    micrographFiles = qtProperty(str, micrographFiles, setMicrographFiles)
    
    @qtSlot()
    def on_micrographFileToolButton_clicked(self):
        '''Called when the user clicks the Pan button.
        '''
        
        files = QtGui.QFileDialog.getOpenFileNames(self, self.tr("Open a set of micrograph images"), self.lastpath)
        if isinstance(files, tuple): files = files[0]
        if len(files) > 0:
            self.lastpath = os.path.dirname(str(files[0]))
            self.setMicrographFiles(files)
    
    def _updateMicrographFiles(self, files=None):
        '''
        '''
        
        self.micrograph_files = files
        if len(files) > 0 and os.path.exists(files[0]):
            total = ndimage_file.count_images(files[0])
            if total > 1:
                self.ui.gainFileComboBox.setEnabled(True)
                self.ui.gainFilePushButton.setEnabled(True)
                
        self.micrographFilesUpdated.emit()
    
    @qtSlot(str)
    def on_micrographComboBox_editTextChanged(self, text):
        '''Called when the user clicks the Pan button.
        '''
        
        _logger.error("on_micrographComboBox_editTextChanged-here1")
        if text == "": return
        files = glob.glob(text)
        if len(files) > 0:
            self.setMicrographFiles(files)
    
    ########################################################################################################################################
    ###### Gaile filename controls
    ########################################################################################################################################
    
    def gainFiles(self):
        '''
        '''
        
        return ",".join(self.gain_files)
    
    def setGainFiles(self, files):
        '''
        '''
        
        _logger.error("setGainFiles-here1")
        try:"+"+files
        except: pass
        else: files = files.split(',')
        self.ui.gainFileComboBox.clear()
        self.ui.gainFileComboBox.addItems(files)
        self._updateGainFiles(files)
        
        self.ui.gainFileComboBox.blockSignals(True)
        if len(files) > 0:
            self.ui.gainFileComboBox.lineEdit().setText(self.ui.gainFileComboBox.itemText(0))
        else:
            self.ui.gainFileComboBox.lineEdit().setText("")
        self.ui.gainFileComboBox.blockSignals(False)
    
    gainFiles = qtProperty(str, gainFiles, setGainFiles)
    
    def _updateGainFiles(self, files=None):
        '''
        '''
        
        self.gain_files = files
        self.gainFilesUpdated.emit()
    
    @qtSlot(str)
    def on_gainFileComboBox_editTextChanged(self, text):
        '''Called when the user clicks the Pan button.
        '''
        
        if text == "": return
        files = glob.glob(text)
        if len(files) > 0:
            self.setGainFiles(files)
    
    @qtSlot()
    def on_gainFilePushButton_clicked(self):
        '''Called when the user clicks the Pan button.
        '''
        
        files = QtGui.QFileDialog.getOpenFileNames(self, self.tr("Open a set of gain images"), self.lastpath)
        if isinstance(files, tuple): files = files[0]
        if len(files) > 0:
            self.lastpath = os.path.dirname(str(files[0]))
            self.setGainFiles(files)
            
    ########################################################################################################################################
    ###### Converted properties
    ########################################################################################################################################
    
    @qtProperty(str)
    def gainFile(self):
        '''
        '''
        
        return self.gain_files[0] if len(self.gain_files) > 0 else ""
    
    @qtProperty(float)
    def window(self):
        '''
        '''
        
        return int(self.ui.windowAngstromDoubleSpinBox.value()/self.ui.pixelSizeDoubleSpinBox.value()) if self.ui.pixelSizeDoubleSpinBox.value() != 0 else 0
        
    ########################################################################################################################################
    
    def setupFineTunePage(self):
        '''
        '''
        
        param = vars(project.default_settings())
        for p in self.parameters:
            param[p]=self.field(p)
        workflow = project.workflow_settings(self.micrographFiles, **param)
        self.ui.monitorWidget.setWorkflow(workflow)
    
    ########################################################################################################################################
    
    ########################################################################################################################################
    
    def showEvent(self, evt):
        '''Window close event triggered - save project and global settings 
        
        :Parameters:
            
        evt : QCloseEvent
              Event for to close the main window
        '''
        
        # Load the settings
        _logger.info("\rLoading project settings ...")
        QtGui.QWizard.showEvent(self, evt)
    
    def idOf(self, page):
        '''
        '''
        
        for id in self.pageIds():
            if self.page(id) == page: return id
        return None
    
    def validateCurrentPage(self):
        '''
        '''
        
        page = self.page(self.currentId())
        if page == self.ui.leginonDBPage:
            return self.ui.leginonWidget.validate()
        elif page == self.ui.manualSettingsPage:
            if self.ui.micrographComboBox.count() == 0: 
                # give error
                return False
            for i in xrange(self.ui.micrographComboBox.count()):
                if not os.path.exists(self.ui.micrographComboBox.itemText(i)): 
                    # give error
                    return False
            return True
        return True

    def onCurrentIDChanged(self, id=None):
        '''
        '''
        
        page = self.page(self.currentId())
        if page is None: return
        page.initializePage()
        if page == self.ui.settingsQuestionPage:
            self.ui.yesLeginonPushButton.setChecked(False)
            self.ui.noLeginonPushButton.setChecked(False)
            button = self.button(QtGui.QWizard.WizardButton.NextButton)
            button.setVisible(False)
        elif page == self.ui.referenceQuestionPage:
            self.ui.yesReferencePushButton.setChecked(False)
            self.ui.noReferencePushButton.setChecked(False)
            button = self.button(QtGui.QWizard.WizardButton.NextButton)
            button.setVisible(False)
        
    def nextId(self):
        '''
        '''
        page = self.page(self.currentId())
        if page == self.ui.manualSettingsPage:
            fields = self.ui.leginonWidget.currentData()
            _logger.error("nextId-here1")
            for key, val in fields.iteritems():
                _logger.error("nextId-here1b-"+str(key))
                self.setField(key, val)
            _logger.error("nextId-here2")
        elif page == self.ui.fineTunePage:
            self.setupFineTunePage()
        elif page == self.ui.monitorPage:
            # save config files
            # - todo: save current state in local directory
            pass
        elif page == self.ui.settingsQuestionPage:
            if self.ui.noLeginonPushButton.isChecked():
                return self.currentId()+2
        elif page == self.ui.referenceQuestionPage:
            if self.ui.noReferencePushButton.isChecked():
                return self.currentId()+2
        return super(MainWindow, self).nextId()
        
    
def module_name(mod):
    ''' Return the name of the module
    
    :Parameters:
    
    mod : module
          Module to format
    
    :Returns:
    
    name : str
           Name of the module
    '''
    
    name = mod.__name__
    idx = name.rfind('.')
    if idx != -1: name = name[idx+1:]
    return name
