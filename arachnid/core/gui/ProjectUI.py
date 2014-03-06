'''

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
from model.ListTableModel import ListTableModel
from HelpUI import Dialog as HelpDialog

from pyui.ProjectUI import Ui_ProjectWizard
from util.qt4_loader import QtGui, qtSlot, QtCore, qtProperty, qtSignal
from arachnid.util import project
from ..image import ndimage_file
from ..metadata import spider_utility
from util import messagebox
from ..parallel import openmp
import logging
import multiprocessing
import arachnid
import os
import glob
import sys
import platform
import functools
sys.setrecursionlimit(10000)


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class MainWindow(QtGui.QWizard):
    '''
    '''
    
    micrographFilesUpdated = qtSignal()
    gainFilesUpdated = qtSignal()
    
    def __init__(self, screen_shot_file=None, parent=None):
        '''
        '''
        
        
        QtGui.QWizard.__init__(self, parent)
        
        self.ui = Ui_ProjectWizard()
        self.ui.setupUi(self)
        #self.subPages={}
        self.lastmicpath = str(QtCore.QDir.currentPath())
        self.lastgainpath = str(QtCore.QDir.currentPath())
        self.micrograph_files = []
        self.gain_files = []
        self.parameters=[]
        self.screen_shot_file=screen_shot_file
        self.helpDialog = HelpDialog(self)
        self.default_spider_path = ' /guam.raid.cluster.software/spider.21.00/'
        
        version = arachnid.__version__
        n=version.find('_')
        if n != -1: version = version[:n]
        self.setWindowTitle("Arachnid - Workflow Creation Wizard - v%s"%version)
        self.docs_url = ""

        
        self.setPixmap(QtGui.QWizard.WatermarkPixmap, QtGui.QPixmap(':/icons/logo/ArachnidLogoWindow_small.png')) #:/icons/icons/icon256x256.png'))
        self.setPixmap(QtGui.QWizard.BackgroundPixmap, QtGui.QPixmap(':/icons/logo/ArachnidLogoWindow_small.png')) #:/icons/icons/icon256x256.png'))
        
        ########################################################################################################################################
        ###### Introduction Page
        ########################################################################################################################################
        self.ui.introductionPage.setTitle('Welcome to Arachnid - v%s'%version)
        if screen_shot_file:
            self.ui.screenShotCheckBox.setCheckState(QtCore.Qt.Checked)
            self.ui.screenShotCheckBox.setEnabled(False)
        
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
        self.ui.leginonWidget = LeginonUI(self, self.helpDialog)
        self.ui.leginonDBLayout.addWidget(self.ui.leginonWidget)
        #self.subPages[self.idOf(self.ui.leginonDBPage)]=self.ui.leginonWidget
        self.ui.leginonWidget.registerPage(self.ui.leginonDBPage)
        #self.ui.leginonWidget.loadFinished.connect(self.next)
        self.ui.leginonWidget.loadFinished.connect(self.onLeginonLoadFinished)
        self.ui.leginonWidget.captureScreen.connect(self.captureScreen)
        
        ########################################################################################################################################
        ###### Reference Page
        ########################################################################################################################################
        self.ui.referenceWidget = ReferenceUI(self, self.helpDialog)
        self.ui.referenceLayout.addWidget(self.ui.referenceWidget)
        #self.subPages[self.idOf(self.ui.referencePage)]=self.ui.referenceWidget    
        self.ui.referenceWidget.registerPage(self.ui.referencePage) 
        #self.ui.referencePage.registerField("referenceEdit*", self.ui.referenceWidget.ui.referenceLineEdit)
        self.ui.referenceWidget.captureScreen.connect(self.captureScreen)
        
        ########################################################################################################################################
        ###### Monitor Page
        ########################################################################################################################################
        self.ui.monitorWidget = MonitorUI(self, self.helpDialog)
        self.ui.monitorLayout.addWidget(self.ui.monitorWidget)
        
        ########################################################################################################################################
        ###### Fine Settings Page
        ########################################################################################################################################
        self.ui.settingsTabWidget = SettingsUI(self)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ui.settingsTabWidget.sizePolicy().hasHeightForWidth())
        self.ui.settingsTabWidget.setSizePolicy(sizePolicy)
        self.ui.settingsHorizontalLayout.addWidget(self.ui.settingsTabWidget)
        
        
        
        self.ui.workflowListView.setModel(self.ui.monitorWidget.model())
        selmodel = self.ui.workflowListView.selectionModel()
        selmodel.currentChanged.connect(self.ui.settingsTabWidget.settingsChanged)
        
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
        self.ui.imageStatTableView.setModel(ListTableModel([], ['Exposure', 'Gain'], ['Width', 'Height', 'Frames', 'Total'], self))
        self.ui.importMessageLabel.setVisible(False)
        
        ########################################################################################################################################
        ###### Additional Settings Page
        ########################################################################################################################################
        
        if os.path.exists(self.default_spider_path):
            self.updateSpiderExe(self.default_spider_path)
        
        self.updateParticleSizeSpinBox = lambda x: self.ui.particleSizeSpinBox.setValue(int(x/self.ui.pixelSizeDoubleSpinBox.value()) if self.ui.pixelSizeDoubleSpinBox.value() > 0 else 0)
        self.updateWindowSizeSpinBox = lambda x: self.ui.windowSizeSpinBox.setValue(int(x/self.ui.pixelSizeDoubleSpinBox.value()) if self.ui.pixelSizeDoubleSpinBox.value() > 0 else 0)
        self.updateMaskDiameterSpinBox = lambda x: self.ui.maskDiameterSpinBox.setValue(int(x/self.ui.pixelSizeDoubleSpinBox.value()) if self.ui.pixelSizeDoubleSpinBox.value() > 0 else 0)
        
        self.updateParticleSizeDoubleSpinBox = lambda x: self.ui.particleSizeDoubleSpinBox.setValue(float(x*self.ui.pixelSizeDoubleSpinBox.value()) if self.ui.pixelSizeDoubleSpinBox.value() > 0 else 0.0)
        self.updateWindowSizeDoubleSpinBox = lambda x: self.ui.windowSizeDoubleSpinBox.setValue(float(x*self.ui.pixelSizeDoubleSpinBox.value()) if self.ui.pixelSizeDoubleSpinBox.value() > 0 else 0.0)
        self.updateMaskDiameterDoubleSpinBox = lambda x: self.ui.maskDiameterDoubleSpinBox.setValue(float(x*self.ui.pixelSizeDoubleSpinBox.value()) if self.ui.pixelSizeDoubleSpinBox.value() > 0 else 0.0)
        
        self.ui.particleSizeDoubleSpinBox.valueChanged.connect(self.updateParticleSizeSpinBox)
        self.ui.windowSizeDoubleSpinBox.valueChanged.connect(self.updateWindowSizeSpinBox)
        self.ui.maskDiameterDoubleSpinBox.valueChanged.connect(self.updateMaskDiameterSpinBox)
        
        self.ui.particleSizeUnitComboBox.currentIndexChanged.connect(self.ui.particleDiameterStackedWidget.setCurrentIndex)
        self.ui.windowSizeUnitComboBox.currentIndexChanged.connect(self.ui.windowSizeStackedWidget.setCurrentIndex)
        self.ui.maskDiameterUnitComboBox.currentIndexChanged.connect(self.ui.maskDiameterStackedWidget.setCurrentIndex)
        
        self.ui.particleSizeUnitComboBox.currentIndexChanged.connect(functools.partial(connect_visible_spin_box, signals=(self.ui.particleSizeDoubleSpinBox.valueChanged, self.ui.particleSizeSpinBox.valueChanged), slots=(self.updateParticleSizeSpinBox, self.updateParticleSizeDoubleSpinBox)))
        self.ui.windowSizeUnitComboBox.currentIndexChanged.connect(functools.partial(connect_visible_spin_box, signals=(self.ui.windowSizeDoubleSpinBox.valueChanged, self.ui.windowSizeSpinBox.valueChanged), slots=(self.updateWindowSizeSpinBox, self.updateWindowSizeDoubleSpinBox)))
        self.ui.maskDiameterUnitComboBox.currentIndexChanged.connect(functools.partial(connect_visible_spin_box, signals=(self.ui.maskDiameterDoubleSpinBox.valueChanged, self.ui.maskDiameterSpinBox.valueChanged), slots=(self.updateMaskDiameterSpinBox, self.updateMaskDiameterDoubleSpinBox)))
        
        self.ui.particleSizeDoubleSpinBox.valueChanged.connect(lambda x: self.ui.windowSizeDoubleSpinBox.setValue(x*1.4))
        self.ui.particleSizeDoubleSpinBox.valueChanged.connect(lambda x: self.ui.maskDiameterDoubleSpinBox.setValue(x*1.2))
  
        self.ui.additionalSettingsPage.registerField(self.param("spider_path*"), self.ui.spiderExecutableLineEdit)
        self.ui.additionalSettingsPage.registerField(self.param("particle_diameter*"), self.ui.particleSizeDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.additionalSettingsPage.registerField(self.param("window_actual*"), self.ui.windowSizeDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.additionalSettingsPage.registerField(self.param("mask_diameter*"), self.ui.maskDiameterDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.additionalSettingsPage.registerField(self.param('worker_count'), self.ui.workerCountSpinBox)
        self.ui.additionalSettingsPage.registerField(self.param('thread_count'), self.ui.threadCountSpinBox)
        self.ui.additionalSettingsPage.registerField(self.param("window"), self, "window")
        self.ui.additionalSettingsPage.registerField(self.param('enable_stderr'), self.ui.enableStderrCheckBox)
        self.ui.additionalSettingsPage.registerField(self.param("disk_mult"), self, 'sampleShape') 
        self.ui.additionalSettingsPage.registerField(self.param("overlap_mult"), self, 'sampleDensity')
        self.ui.additionalSettingsPage.registerField(self.param("threshold_minimum"), self, 'sampleDensityMin')
        self.ui.sampleShapeComboBox.setItemData(0, 0.6)
        self.ui.sampleShapeComboBox.setItemData(1, 0.35)
        self.ui.sampleDensityComboBox.setItemData(0, 1.0)
        self.ui.sampleDensityComboBox.setItemData(1, 1.2)
        self.ui.sampleDensityComboBox.setItemData(2, 0.8)
        
        thread_count = 1
        if openmp.is_openmp_enabled():
            thread_count = openmp.get_num_procs()
        else:
            try: thread_count=multiprocessing.cpu_count()
            except: pass
        self.ui.workerCountSpinBox.setValue(thread_count)
        self.ui.selectLeginonInformationLabel.setVisible(False)
        self.ui.selectReferenceInformationLabel.setVisible(False)
    
    @qtSlot()
    def on_documentationURLToolButton_clicked(self):
        '''
        '''
        
        model = self.ui.workflowListView.model()
        selmodel = self.ui.workflowListView.selectionModel()
        index = selmodel.currentIndex()
        if index is None: return
        program = model.data(index, QtCore.Qt.UserRole)
        if program is None: return
        option = self.ui.settingsTabWidget.selectedOption()
        
        if self.docs_url == "":
            url, ok = QtGui.QInputDialog.getText(self, 'Use default URL?', 'URL:', QtGui.QLineEdit.Normal, 'code.google.com/p')
            if ok:
                self.docs_url = 'http://'+url+'/arachnid/docs/api_generated/'
            else: return
        if option is not None:
            QtGui.QDesktopServices.openUrl(self.docs_url+program.id()+".html#cmdoption-%s%s"%(program.program_name(), option))
        else:
            QtGui.QDesktopServices.openUrl(self.docs_url+program.id()+".html")
    
    
    @qtSlot()
    def on_spiderExecutablePushButton_clicked(self):
        '''
        
        Possible names:
            - spider_linux  spider_linux_mp_intel  spider_linux_mp_intel64  spider_linux_mpi_opt64  spider_linux_mp_opt64  spider_osx_32  spider_osx_64
        '''
        
        file_path = QtGui.QFileDialog.getExistingDirectory(self, self.tr("Directory containing SPIDER executables"))
        if file_path == "": return
        self.updateSpiderExe(file_path)
    
    def updateSpiderExe(self, file_path):
        '''
        '''
        
        exe = project.determine_spider(file_path)
        if exe == "":
            messagebox.error_message(self, "Cannot find SPIDER executables in %s"%file_path)
        self.ui.spiderExecutableLineEdit.setText(exe)
    
    @qtSlot()
    def on_spiderExecutableToolButton_clicked(self):
        '''
        '''
        
        self.helpDialog.setHTML(self.ui.spiderExecutableLineEdit.toolTip())
        self.helpDialog.show()
    
    @qtSlot()
    def on_sampleShapeInfoToolButton_clicked(self):
        '''
        '''
        
        self.helpDialog.setHTML(self.ui.sampleShapeComboBox.toolTip())
        self.helpDialog.show()
    
    @qtSlot()
    def on_sampleDensityInfoToolButton_clicked(self):
        '''
        '''
        
        self.helpDialog.setHTML(self.ui.sampleDensityComboBox.toolTip())
        self.helpDialog.show()
    
    @qtSlot()
    def on_selectReferenceInformationToolButton_clicked(self):
        '''
        '''
        
        self.helpDialog.setHTML(self.ui.yesReferencePushButton.toolTip()+self.ui.noReferencePushButton.toolTip())
        self.helpDialog.show()
        #self.ui.selectReferenceInformationLabel.setVisible(self.ui.selectReferenceInformationToolButton.isChecked())
    
    @qtSlot()
    def on_selectLeginonInformationToolButton_clicked(self):
        '''
        '''
        
        self.helpDialog.setHTML(self.ui.yesLeginonPushButton.toolTip()+self.ui.noLeginonPushButton.toolTip())
        self.helpDialog.show()
        
        #self.ui.selectLeginonInformationLabel.setVisible(self.ui.selectLeginonInformationToolButton.isChecked())
    
    @qtSlot()
    def on_settingsInformatToolButton_clicked(self):
        '''
        '''
        
        self.helpDialog.setHTML(self.ui.workflowListView.toolTip())
        self.helpDialog.show()
    
    @qtSlot()
    def on_dimensionInformationToolButton_clicked(self):
        '''
        '''
        
        self.helpDialog.setHTML(self.ui.particleSizeDoubleSpinBox.toolTip())
        self.helpDialog.show()
    
    @qtSlot()
    def on_parallelInformationToolButton_clicked(self):
        '''
        '''
        
        self.helpDialog.setHTML(self.ui.workerCountSpinBox.toolTip())
        self.helpDialog.show()
    
    @qtSlot()
    def on_logInformationToolButton_clicked(self):
        '''
        '''
        
        self.helpDialog.setHTML(self.ui.enableStderrCheckBox.toolTip())
        self.helpDialog.show()
    
    @qtSlot()
    def on_micrographInformationToolButton_clicked(self):
        '''
        '''
        
        self.helpDialog.setHTML(self.ui.micrographComboBox.toolTip())
        self.helpDialog.show()
    
    @qtSlot()
    def on_gainInformationToolButton_clicked(self):
        '''
        '''
        
        self.helpDialog.setHTML(self.ui.gainFileComboBox.toolTip())
        self.helpDialog.show()
    
    def param(self, name):
        '''
        '''
        
        if name[-1]=='*':
            self.parameters.append(name[:len(name)-1])
        else:
            self.parameters.append(name)
        
        return name
    
    ########################################################################################################################################
    ###### Sample controls
    ########################################################################################################################################
    
    def sampleShape(self):
        '''
        '''
        
        if self.ui.sampleShapeComboBox.count() == 0 or self.ui.sampleShapeComboBox.currentIndex() < 1: return 0.6
        return float(self.ui.sampleShapeComboBox.itemData(self.ui.sampleShapeComboBox.currentIndex()))
    sampleShape = qtProperty(float, sampleShape)
    
    def sampleDensity(self):
        '''
        '''
        
        if self.ui.sampleDensityComboBox.count() == 0 or self.ui.sampleDensityComboBox.currentIndex() < 1: return 1.0
        return float(self.ui.sampleDensityComboBox.itemData(self.ui.sampleDensityComboBox.currentIndex()))
    sampleDensity = qtProperty(float, sampleDensity)
    
    def sampleDensityMin(self):
        '''
        '''
        
        if self.ui.sampleDensityComboBox.count() == 0 or self.ui.sampleDensityComboBox.currentIndex() < 1: return 25
        return int(10 if self.ui.sampleDensityComboBox.currentIndex() == 1 else 25)
    sampleDensityMin = qtProperty(int, sampleDensityMin)

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
        
        try:"+"+files
        except: pass
        else: files = files.split(',')
        self.ui.micrographComboBox.blockSignals(True)
        self.ui.micrographComboBox.clear()
        self.ui.micrographComboBox.addItems(files)
        self._updateMicrographFiles(files)
        
        if len(files) > 0:
            self.lastmicpath = os.path.dirname(self.ui.micrographComboBox.itemText(0))
            self.ui.micrographComboBox.lineEdit().setText(self.ui.micrographComboBox.itemText(0))
        else:
            self.ui.micrographComboBox.lineEdit().setText("")
        self.ui.micrographComboBox.blockSignals(False)
        self.updateImageInfo()
    
    micrographFiles = qtProperty(str, micrographFiles, setMicrographFiles)
    
    @qtSlot()
    def on_micrographFileToolButton_clicked(self):
        '''Called when the user clicks the Pan button.
        '''
        
        
        files = QtGui.QFileDialog.getOpenFileNames(self, self.tr("Open a set of micrograph images"), self.lastmicpath)
        if isinstance(files, tuple): files = files[0]
        if len(files) > 0:
            self.lastmicpath = os.path.dirname(str(files[0]))
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
        
        if text == "": return
        files = glob.glob(text)
        if len(files) > 0:
            self.setMicrographFiles(files)
    
    ########################################################################################################################################
    ###### Gain filename controls
    ########################################################################################################################################
    
    def gainFiles(self):
        '''
        '''
        
        return ",".join(self.gain_files)
    
    def setGainFiles(self, files):
        '''
        '''
        
        try:"+"+files
        except: pass
        else: files = files.split(',')
        files = list(set(files))
        self.ui.gainFileComboBox.blockSignals(True)
        self.ui.gainFileComboBox.clear()
        self.ui.gainFileComboBox.addItems(files)
        self._updateGainFiles(files)
        
        if len(files) > 0:
            self.lastgainpath = os.path.dirname(self.ui.gainFileComboBox.itemText(0))
            self.ui.gainFileComboBox.lineEdit().setText(self.ui.gainFileComboBox.itemText(0))
        else:
            self.ui.gainFileComboBox.lineEdit().setText("")
        self.ui.gainFileComboBox.blockSignals(False)
        self.updateImageInfo()
    
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
        
        files = QtGui.QFileDialog.getOpenFileNames(self, self.tr("Open a set of gain images"), self.lastgainpath)
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
    
    @gainFile.setter
    def setGainFile(self, val):
        '''
        '''
        
        self.gain_files=[val]
    
    @qtProperty(int)
    def window(self):
        '''
        '''
        
        return int(self.ui.windowSizeDoubleSpinBox.value()/self.ui.pixelSizeDoubleSpinBox.value()) if self.ui.pixelSizeDoubleSpinBox.value() != 0 else 0
    
    @window.setter
    def setWindow(self, val):
        '''
        '''
        
        self.ui.windowSizeDoubleSpinBox.setValue(val*self.ui.pixelSizeDoubleSpinBox.value())
        
        
    ########################################################################################################################################
    
    def setupFineTunePage(self):
        '''
        '''
        
        param = {}
        for p in self.parameters:
            val = self.field(p)
            if p == 'input_files': val = val.split(',')
            param[p]=val
        workflow = project.workflow_settings(self.micrograph_files, param)
        self.ui.monitorWidget.setLogFile('project.log')
        self.ui.monitorWidget.setWorkflow(workflow)
    
    def saveFineTunePage(self):
        '''
        '''
        
        self.ui.monitorWidget.saveState()
        project.write_workflow(self.ui.monitorWidget.workflow())
        #todo add update if project option changes!
        
    def loadProject(self):
        '''
        '''
        
        settings = project.default_settings()
        if len(settings.input_files) == 0: return
        
        workflow = project.workflow_settings(settings.input_files, vars(settings))
        self.ui.monitorWidget.setWorkflow(workflow)
        for p in self.parameters:
            if not hasattr(workflow[0].values, p): continue
            val = getattr(workflow[0].values, p)
            if p == 'input_files':
                val = ",".join(val)
            self.setField(p, val)
        print 'load:', self.idOf(self.ui.fineTunePage)
        self.setStartId(self.idOf(self.ui.fineTunePage))
        self.restart()
        self.next()
        
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
        self.loadProject()
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
        if page == self.ui.introductionPage:
            if self.ui.screenShotCheckBox.checkState() == QtCore.Qt.Checked and not self.screen_shot_file:
                self.screen_shot_file = "data/screen_shots/wizard_screen_shot_0000.png"
        elif page == self.ui.leginonDBPage:
            if not self.ui.leginonWidget.validate():
                return False
        elif page == self.ui.manualSettingsPage:
            if self.ui.micrographComboBox.count() == 0: 
                messagebox.error_message(self, "No micrographs to process!")
                # give error
                return False
            for i in xrange(self.ui.micrographComboBox.count()):
                if not os.path.exists(self.ui.micrographComboBox.itemText(i)): 
                    messagebox.error_message(self, "Micrograph not found!", self.ui.micrographComboBox.itemText(i))
                    return False
        self.captureScreen()
        return True
    
    def captureScreen(self, subid=0):
        '''
        '''
        
        if self.screen_shot_file:
            if os.path.dirname(self.screen_shot_file) != "" and not os.path.exists(os.path.dirname(self.screen_shot_file)):
                try:os.makedirs(os.path.dirname(self.screen_shot_file))
                except: pass
            originalPixmap = QtGui.QPixmap.grabWidget(self)
            screen_shot=spider_utility.spider_filename(self.screen_shot_file, self.currentId()*10+subid)
            originalPixmap.save(os.path.splitext(screen_shot)[0]+'.png', 'png')

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
            
    def initializePage(self, id):
        '''
        '''
        
        page = self.page(id)
        if page == self.ui.fineTunePage:
            self.setupFineTunePage()
        elif page == self.ui.monitorPage:
            self.saveFineTunePage()
        
    def nextId(self):
        '''
        '''
        
        page = self.page(self.currentId())
        if page == self.ui.settingsQuestionPage:
            if self.ui.noLeginonPushButton.isChecked():
                return self.currentId()+2
        elif page == self.ui.referenceQuestionPage:
            if self.ui.noReferencePushButton.isChecked():
                return self.currentId()+2
        return super(MainWindow, self).nextId()
    
    def onLeginonLoadFinished(self):
        '''
        '''
        
        fields = self.ui.leginonWidget.currentData()
        if fields is None:
            messagebox.error_message(self, "Can only process one pixel size at a time!")
            return
        for key, val in fields.iteritems():
            self.setField(key, val)
        self.ui.importMessageLabel.setVisible(True)
        
    def updateImageInfo(self):
        '''
        '''
        self.ui.manualSettingsPage.updateGeometry()
        
        mics = self.micrograph_files
        if len(mics) == 0: return
        header = ndimage_file.read_header(mics[0])
        data = [
                [header['nx'], 'N/A'],
                [header['ny'], 'N/A'],
                #[header['nz'], 'N/A'],
                [header['count'], 'N/A'],
                [len(mics), 'N/A'],
                ]
        mics = self.gain_files
        if len(mics) > 0:
            header = ndimage_file.read_header(mics[0])
            total = len(set(mics))
            data[0][1] = header['nx']
            data[1][1] = header['ny']
            data[2][1] = header['count']
            data[3][1] = total
        self.ui.imageStatTableView.model().setData(data)
            
        
def connect_visible_spin_box(index, signals, slots):
    '''
    '''
    prev = 0 if index == 1 else 1
    signals[prev].disconnect(slots[prev])
    signals[index].connect(slots[index])


