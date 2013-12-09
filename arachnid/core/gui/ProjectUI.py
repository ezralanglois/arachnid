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

from pyui.ProjectUI import Ui_ProjectWizard
from util.qt4_loader import QtGui, qtSlot, QtCore, qtSignal
import logging
import multiprocessing
import arachnid
import os


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class MainWindow(QtGui.QWizard):
    '''
    '''
    
    def __init__(self, parent=None):
        '''
        '''
        
        
        QtGui.QWizard.__init__(self, parent)
        
        self.ui = Ui_ProjectWizard()
        self.ui.setupUi(self)
        self.subPages={}
        self.lastpath = str(QtCore.QDir.currentPath())
        
        
        version = arachnid.__version__
        n=version.find('_')
        if n != -1: version = version[:n]
        self.ui.introductionPage.setTitle('Welcome to Arachnid - v%s'%version)
        self.setWindowTitle("Arachnid - Workflow Creation Wizard - v%s"%version)

        
        self.setPixmap(QtGui.QWizard.WatermarkPixmap, QtGui.QPixmap(':/icons/icons/icon256x256.png'))
        self.setPixmap(QtGui.QWizard.BackgroundPixmap, QtGui.QPixmap(':/icons/icons/icon256x256.png'))
        
        self.currentIdChanged.connect(self.onCurrentIDChanged)
        self.ui.yesLeginonPushButton.clicked.connect(self.next)
        self.ui.noLeginonPushButton.clicked.connect(self.next)
        self.ui.yesReferencePushButton.clicked.connect(self.next)
        self.ui.noReferencePushButton.clicked.connect(self.next)
        
        self.ui.leginonWidget = LeginonUI()
        self.ui.leginonDBLayout.addWidget(self.ui.leginonWidget)
        self.subPages[self.idOf(self.ui.leginonDBPage)]=self.ui.leginonWidget
        
        self.ui.referenceWidget = ReferenceUI()
        self.ui.referenceLayout.addWidget(self.ui.referenceWidget)
        self.subPages[self.idOf(self.ui.leginonDBPage)]=self.ui.referenceWidget        
        self.ui.referencePage.registerField("referenceEdit*", self.ui.referenceWidget.ui.referenceLineEdit)
        
        self.ui.monitorWidget = MonitorUI()
        self.ui.monitorLayout.addWidget(self.ui.monitorWidget)
        
        self.ui.manualSettingsPage.registerField("pixelSize*", self.ui.pixelSizeDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.manualSettingsPage.registerField("voltage*", self.ui.voltageDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.manualSettingsPage.registerField("cs*", self.ui.csDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.manualSettingsPage.registerField("micrograph*", self.ui.micFileLineEdit)
        
        #self.ui.manualSettingsPage.registerField("gain*", self.ui.gainFileLineEdit) - add/remove dynamiically!
        
        self.ui.particleSizeDoubleSpinBox.valueChanged.connect(lambda x: self.ui.particleDiameterPixelLabel.setText("%d (Px)"%(int(x/self.ui.pixelSizeDoubleSpinBox.value()))) if self.ui.pixelSizeDoubleSpinBox.value() > 0 else None)
        self.ui.windowAngstromDoubleSpinBox.valueChanged.connect(lambda x: self.ui.windowWidthAnstromsLabel.setText("%d (Px)"%(x/self.ui.pixelSizeDoubleSpinBox.value())) if self.ui.pixelSizeDoubleSpinBox.value() > 0 else None)
        self.ui.maskDiameterDoubleSpinBox.valueChanged.connect(lambda x: self.ui.maskDiameterPixelLabel.setText("%d (Px)"%(int(x/self.ui.pixelSizeDoubleSpinBox.value()))) if self.ui.pixelSizeDoubleSpinBox.value() > 0 else None)
        self.ui.particleSizeDoubleSpinBox.valueChanged.connect(lambda x: self.ui.windowAngstromDoubleSpinBox.setValue(x*1.4))
        self.ui.particleSizeDoubleSpinBox.valueChanged.connect(lambda x: self.ui.maskDiameterDoubleSpinBox.setValue(x*1.2))
        self.ui.additionalSettingsPage.registerField("particleSize*", self.ui.particleSizeDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.additionalSettingsPage.registerField("windowAngstrom*", self.ui.windowAngstromDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        self.ui.additionalSettingsPage.registerField("maskDiameter*", self.ui.maskDiameterDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
        
        
        
        try:
            self.ui.workerCountSpinBox.setValue(multiprocessing.cpu_count())
        except: pass
        
        self.map_param_to_widget = dict(apix=self.ui.pixelSizeDoubleSpinBox,
                                        voltage=self.ui.voltageDoubleSpinBox,
                                        cs=self.ui.csDoubleSpinBox,
                                        window_ang=self.ui.windowAngstromDoubleSpinBox,
                                        particle_diameter=self.ui.particleSizeDoubleSpinBox,
                                        mask_diameter=self.ui.maskDiameterDoubleSpinBox,
                                        worker_count=self.ui.workerCountSpinBox,
                                        thread_count=self.ui.threadCountSpinBox,
                                        invert=self.ui.invertCheckBox,
                                        gain_reference_file=self.ui.gainFileLineEdit.text(),
                                        )
    
    def idOf(self, page):
        '''
        '''
        
        for id in self.pageIds():
            if self.page(id) == page: return id
        return None
    """
    def validateCurrentPage(self):
        '''
        '''
        
        #if self.currentId() in self.subPages:
        #    return self.subPages[self.currentId()].validateCurrentPage()
        
        page = self.page(self.currentId())
        if page is None: return
        if page == self.ui.settingsQuestionPage or \
           page == self.ui.referenceQuestionPage:
            return self.ui.yesLeginonPushButton.isChecked() or self.ui.noLeginonPushButton.isChecked() 
        return True
    """
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
        
        if self.page(self.currentId()) == self.ui.settingsQuestionPage:
            if self.ui.noLeginonPushButton.isChecked():
                return self.currentId()+2
        elif self.page(self.currentId()) == self.ui.referenceQuestionPage:
            if self.ui.noReferencePushButton.isChecked():
                return self.currentId()+2
        return super(MainWindow, self).nextId()
    
    @qtSlot()
    def on_micrographFileToolButton_clicked(self):
        '''Called when the user clicks the Pan button.
        '''
        
        files = QtGui.QFileDialog.getOpenFileNames(self, self.tr("Open a set of micrograph images"), self.lastpath)
        if isinstance(files, tuple): files = files[0]
        if len(files) > 0:
            self.lastpath = os.path.dirname(str(files[0]))
            #self.openMicrograph(files)
        
        
