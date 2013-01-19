''' Main window display for the plotting tool

.. Created on Dec 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from pyui.ProjectWizard import Ui_ProjectWizard, _fromUtf8
from PyQt4 import QtCore
from PyQt4 import QtGui

from ..property import pyqtProperty
from .. import property
from .. import ndimage_file, ndimage_utility, spider_utility
#from .. import format, format_utility, analysis, ndimage_file, ndimage_utility, spider_utility
import logging, numpy, os, glob, multiprocessing, subprocess, psutil
from arachnid.pyspider import project

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class MainWindow(QtGui.QWizard):
    ''' Main window display for the plotting tool
    '''
    
    def __init__(self, parent=None):
        "Initialize a basic viewer window"
        
        QtGui.QWizard.__init__(self, parent)
        self.ui = Ui_ProjectWizard()
        self.ui.setupUi(self)
        self.lastpath = str(QtCore.QDir.currentPath())
        self.jobUpdateTimer = QtCore.QTimer(self)
        self.jobUpdateTimer.setInterval(500)
        self.jobUpdateTimer.setSingleShot(False)
        self.scripts = []
        self.current_script_index=0
        self.current_running=0
        self.modules = []
        self.fullparam = {}
        self.config_path = {}
        self.micrograph = None
        self.progress_file=""
        self.output = '.'
        self.proc = None
        self.fine_param={}
        self.leginon_filename="mapped_micrographs/mic_0000000"
        self.inifile = 'ara_project.ini'
        self.param = {'orig_files': '', 'input_files': '', 'is_film':False, 
                      'curr_apix':0.0, 'raw_reference':'',
                      'apix': 0.0, 'voltage': 0.0, 'particle_diameter': 0.0, 'cs': 0.0,
                      'worker_count': 0, 'thread_count': 1, 'window_size': 0, 'ext': 'dat'
                     }
        self.appparam = {'page': 0, 'pid': -1, 'pid_time': -1} # save run state - pid - test if exists
        self.settings = []
        self.settings.append(('Global', self.param))
        self.settings.append(('App', self.appparam))
        
        # save fine tuned options
        
        self.connect(self, QtCore.SIGNAL("currentIdChanged(int)"), self.onPageChanged)
        self.connect(self.ui.referenceLineEdit, QtCore.SIGNAL("editingFinished()"), self.onReferenceEditChanged)
        self.connect(self.ui.micrographFileLineEdit, QtCore.SIGNAL("editingFinished()"), self.onMicrographLineEditChanged)
        self.connect(self.ui.extensionLineEdit, QtCore.SIGNAL("editingFinished()"), self.onExtensionEditChanged)
        self.connect(self.jobUpdateTimer, QtCore.SIGNAL("timeout()"), self.onMonitorUpdate  )
        
        #WatermarkPixmap, LogoPixmap, BannerPixmap, BackgroundPixmap
        self.setPixmap(QtGui.QWizard.WatermarkPixmap, QtGui.QPixmap(_fromUtf8(':/icons/icons/icon256x256.png')))
        self.setPixmap(QtGui.QWizard.BackgroundPixmap, QtGui.QPixmap(_fromUtf8(':/icons/icons/icon256x256.png')))
        #self.setPixmap(QtGui.QWizard.BannerPixmap, QtGui.QPixmap(_fromUtf8(':/icons/icons/icon64x64.png')))
        #self.setPixmap(QtGui.QWizard.LogoPixmap, QtGui.QPixmap(_fromUtf8(':/icons/icons/icon64x64.png')))
        
        self.loadSettings()
        
        #Page 1
        self.ui.micrographFileLineEdit.setText(self.param['input_files'])
        self.param['input_files']=""
        if self.param['is_film']:
            self.ui.invertCheckBox.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.ui.invertCheckBox.setCheckState(QtCore.Qt.Checked)
        self.openMicrograph(self.micrographFiles('orig_files'))
        
        #Page 2
        self.ui.referenceLineEdit.setText(self.param['raw_reference'])
        self.ui.referencePixelSizeDoubleSpinBox.setValue(self.param['curr_apix'])
        self.openReference(self.param['raw_reference'])
        
        #Page 3
        self.ui.pixelSizeDoubleSpinBox.setValue(self.param['apix'])
        self.ui.voltageDoubleSpinBox.setValue(self.param['voltage'])
        self.ui.particleSizeDoubleSpinBox.setValue(self.param['particle_diameter'])
        self.ui.csDoubleSpinBox.setValue(self.param['cs'])
        
        #Page 4
        self.ui.workerCountSpinBox.setValue(self.param['worker_count'])
        if self.ui.workerCountSpinBox.value() == 0:
            try:
                self.ui.workerCountSpinBox.setValue(multiprocessing.cpu_count())
            except: pass
        self.ui.threadCountSpinBox.setValue(self.param['thread_count'])
        self.ui.windowSizeSpinBox.setValue(self.param['window_size'])
        self.ui.extensionLineEdit.setText(self.param['ext'])
        
        #Page 5
        
        #Page 6
        
        self.jobListModel = QtGui.QStandardItemModel(self)
        self.ui.jobListView.setModel(self.jobListModel)
        
        self.jobStatusIcons=[]
        for filename in [':/mini/mini/clock.png', ':/mini/mini/arrow_refresh.png', ':/mini/mini/tick.png', ':/mini/mini/cross.png']:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(_fromUtf8(filename)), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.jobStatusIcons.append(icon)
        self.startMonitor()
        
    
    def saveSettings(self):
        ''' Save the settings of the controls in the settings map
        '''
        
        settings = QtCore.QSettings(self.inifile, QtCore.QSettings.IniFormat)
        for group, param in self.settings:
            settings.beginGroup(group)
            for name, method in param.iteritems():
                settings.setValue(name, QtCore.QVariant(method))
            settings.endGroup()
    
    def loadSettings(self):
        ''' Load the settings of controls specified in the settings map
        '''
        
        settings = QtCore.QSettings(self.inifile, QtCore.QSettings.IniFormat)
        for group, param in self.settings:
            settings.beginGroup(group)
            for name, val in param.iteritems():
                sval = str(settings.value(name).toString())
                if sval != "":
                    if sval.lower()=='false': sval=False
                    param[name] = val.__class__(sval)
            settings.endGroup()
            
        
    def closeEvent(self, evt):
        '''Window close event triggered - save project and global settings 
        
        :Parameters:
            
        evt : QCloseEvent
              Event for to close the main window
        '''
        
        self.appparam['page'] = self.currentId()
        self.saveSettings()
        QtGui.QMainWindow.closeEvent(self, evt)
    
    def onPageChanged(self, page=None):
        '''
        '''
        
        if page is None: page = self.currentId()
        enable=True
        if page == 0:
            for i in xrange(self.appparam['page']): 
                self.next()
            l = self.appparam['page']
            self.appparam['page']=0
            if l > 0: return
        elif page == 1:
            files = self.micrographFiles('orig_files')
            if len(files)==0: enable=False
            if project.legion_to_spider.is_legion_filename(files):
                data_ext = self.param['ext']
                if data_ext[0]!='.': data_ext = '.'+data_ext
                leginon_filename = os.path.splitext(self.leginon_filename)[0]+data_ext
                files = project.legion_to_spider.convert_to_spider(files, leginon_filename, 0)
                _logger.info("Added %d new files of %d"%(len(files), len(self.param['orig_files'])))
                self.param['input_files'] = ','.join(compress_filenames(files))
            
            for filename in files:
                filename = str(filename)
                if not os.path.exists(filename) or not spider_utility.is_spider_filename(filename):
                    enable=False
                    break
        elif page == 2:
            enable = self.param['curr_apix'] > 0 and self.param['raw_reference'] != "" and os.path.exists(self.param['raw_reference'])
        elif page == 3:
            enable = self.param['apix'] > 0 and self.param['voltage'] > 0 and self.param['particle_diameter'] > 0 and self.param['cs'] > 0
        elif page == 4:
            enable = len(self.param['ext']) == 3
        elif page == 5:
            self.button(QtGui.QWizard.NextButton).setText('Generate')
            self.setupAdvancedSettings()
        elif page == 6:
            self.setupJobSettings()
        self.button(QtGui.QWizard.NextButton).setEnabled(enable)
    
    ####################################################################################
    #
    # Page 7 Controls, discreet struct fall 2012, lessons, homeworks, add content, file, (last homework 8 solutions)
    #                  communicate, quick message, groups, all students -> homework solutions for last homework posted on angel, under homeworks
    #
    ####################################################################################
    
    
    @QtCore.pyqtSlot(name='on_runJobButton_clicked')
    def onRunJob(self):
        '''
        '''
        
        self.proc = subprocess.Popen(["/bin/sh", os.path.abspath(self.scripts[self.current_script_index][0])])
        self.appparam['pid']=self.proc.pid
        self.appparam['pid_time'] = int(psutil.Process(self.proc.pid).create_time)
        self.startMonitor()
    
    def startMonitor(self):
        '''
        '''
        
        if not self.checkRunningState(): return
        self.ui.jobProgressBar.setValue(0)
        self.jobUpdateTimer.start()
    
    def onMonitorUpdate(self):
        ''' Invoked by the timer
        '''
        
        self.updateCurrentRunning()
        lines = self.updateLogTextEdit()
        
        
        #tag = 'Finished: '
        progress, maximum = self.getProgress(lines)
        
        if maximum is not None:
            if self.ui.jobProgressBar.maximum() != maximum:
                self.ui.jobProgressBar.setMaximum(maximum)
            self.ui.jobProgressBar.setValue(progress+1)
        
        if not self.checkRunningState():
            self.jobUpdateTimer.stop()
            if self.current_running < len(self.modules):
                self.updateCurrentRunning()
                lines = self.updateLogTextEdit()
                completed = lines[-1].find('Completed') if len(lines) > 0 else -1
                if completed > -1:
                    self.jobListModel.item(self.current_running).setIcon(self.jobStatusIcons[2])
                else:
                    self.jobListModel.item(self.current_running).setIcon(self.jobStatusIcons[3])
            return
        
    def getProgress(self, lines, tag='Finished: '):
        '''
        '''
        
        progress, maximum = None, None
        for i in xrange(len(lines)-1, -1, -1):
            line = lines[i]
            idx = line.find(tag)
            if idx != -1:
                line = line[idx+len(tag):]
                idx = line.find(' -')
                if idx != -1: line = line[:idx]
                progress, maximum = tuple([int(v) for v in line.split(',')])
                break
        return progress, maximum
        
    def updateLogTextEdit(self):
        ''' Update the log text edit for the current process
        '''
        
        idx = self.current_running
        filename = str(self.jobListModel.item(idx).data(QtCore.Qt.UserRole).toString())
        try:
            fin = open(filename)
        except: return []
        try:
            lines = fin.readlines()
        except:
            return []
        finally:
            fin.close()
        lines.reverse()
        self.ui.logTextEdit.setPlainText(''.join(lines))
        
        #QTextCursor c =  GUI.TextEditStatus->textCursor();
        #c.movePosition(QTextCursor::End);
        #GUI.TextEditStatus->setTextCursor(c);
        
        lines.reverse()
        return lines
    
    def updateCurrentRunning(self):
        '''
        '''
        
        fin = open(self.scripts[self.current_script_index][1])
        lines = fin.readlines()
        fin.close()
        self.current_running=len(lines)
        for i in xrange(self.current_running):
            self.jobListModel.item(i).setIcon(self.jobStatusIcons[2])
        self.jobListModel.item(self.current_running).setIcon(self.jobStatusIcons[1])
        
    def checkRunningState(self):
        '''
        '''
        
        running = self.isProcessRunning()
        self.ui.runJobButton.setChecked(running)
        self.ui.runJobButton.setEnabled(not running)
        self.button(QtGui.QWizard.BackButton).setEnabled(not running)
        return running
        
    def isProcessRunning(self):
        ''' Test is the current process is running
        '''
        
        if self.proc is not None and self.proc.poll() is not None: return False
        #print self.appparam['pid']
        if self.appparam['pid_time'] < 0: return False
        if self.appparam['pid'] > 0:
            try:
                p = psutil.Process(self.appparam['pid'])
            except psutil.error.NoSuchProcess:
                curr_time = -1
            else: curr_time = int(p.create_time)
        else: curr_time = -1
        #print self.appparam['pid'], self.appparam['pid_time']
        #print self.appparam['pid'], curr_time, self.appparam['pid_time'], (curr_time-self.appparam['pid_time'])
        return curr_time == self.appparam['pid_time']
        
    def setupJobSettings(self):
        ''' Setup the job monitor page
        '''
            
        for mod, param in self.modules:
            param.update( vars(self.fine_param[mod]) )
        self.scripts = project.write_config(self.modules, self.fullparam, self.config_path, self.output, **self.param)
        self.jobListModel.clear()
        for mod, param in self.modules:
            item = QtGui.QStandardItem(self.jobStatusIcons[0], project.module_name(mod))
            item.setData(param['log_file'], QtCore.Qt.UserRole)
            self.jobListModel.appendRow(item)
        
            
    ####################################################################################
    #
    # Page 5 Controls
    #
    ####################################################################################
    
    def setupAdvancedSettings(self):
        ''' Setup the advanced settings page
        '''
        
        id_len = spider_utility.spider_id_length(os.path.splitext(self.micrographFiles()[0])[0])
        if id_len == 0: raise ValueError, "Input file not a SPIDER file - id length 0"
        self.modules, self.fullparam, self.config_path = project.workflow(output=self.output, id_len=id_len, **self.param)
        
        self.ui.treeViews = []
        self.ui.settingsTabWidget.clear()
        for mod, param in self.modules:
            extra=dict()
            extra.update(self.param)
            extra.update(param)
            treeView = QtGui.QTreeView()
            property.setView(treeView)
            self.ui.settingsTabWidget.addTab(treeView, project.module_name(mod))
            root, values = project.program.generate_settings(mod, pyqtProperty.PyqtProperty, QtCore.QObject, description="", **extra)
            self.fine_param[mod]=values
            treeView.model().addItem(root)
            width = treeView.model().maximumTextWidth(self.fontMetrics(), treeView.indentation())
            treeView.setColumnWidth(0, width)
            treeView.expandAll()
            self.ui.treeViews.append(treeView)
    
    ####################################################################################
    #
    # Page 4 Controls
    #
    ####################################################################################  
    
    @QtCore.pyqtSlot('int', name='on_workerCountSpinBox_valueChanged')
    def onWorkerCountChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['worker_count'] = value
    
    @QtCore.pyqtSlot('int', name='on_threadCountSpinBox_valueChanged')
    def onThreadCountChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['thread_count'] = value
    
    @QtCore.pyqtSlot('int', name='on_windowSizeSpinBox_valueChanged')
    def onWindowSizeChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['window_size'] = value
    
    @QtCore.pyqtSlot(name='on_extensionLineEdit_editingFinished)')
    def onExtensionEditChanged(self):
        '''
        '''
        
        text = str(self.ui.extensionLineEdit.text())
        self.param['ext'] = text
        self.onPageChanged()
            
    ####################################################################################
    #
    # Page 3 Controls
    #
    ####################################################################################  
    
    @QtCore.pyqtSlot('double', name='on_pixelSizeDoubleSpinBox_valueChanged')
    def onPixelSizeChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['apix'] = value
        if self.param['apix'] > 0:
            self.ui.windowSizeSpinBox.blockSignals(True)
            val = int(self.param['particle_diameter']/self.param['apix']*1.3)
            val += (val%2)
            self.ui.windowSizeSpinBox.setValue(val)
            self.ui.windowSizeSpinBox.blockSignals(False)
        self.onPageChanged()
        
    @QtCore.pyqtSlot('double', name='on_voltageDoubleSpinBox_valueChanged')
    def onVoltageChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['voltage'] = value
        self.onPageChanged()
        
    @QtCore.pyqtSlot('double', name='on_particleSizeDoubleSpinBox_valueChanged')
    def onParticleSizeChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['particle_diameter'] = value
        if self.param['apix'] > 0:
            self.ui.windowSizeSpinBox.blockSignals(True)
            self.ui.windowSizeSpinBox.setValue(self.param['particle_diameter']/self.param['apix']*1.3)
            self.ui.windowSizeSpinBox.blockSignals(False)
        self.onPageChanged()
        
    @QtCore.pyqtSlot('double', name='on_csDoubleSpinBox_valueChanged')
    def onCsChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['cs'] = value
        self.onPageChanged()
            
    ####################################################################################
    #
    # Page 2 Controls
    #
    ####################################################################################    
    
    @QtCore.pyqtSlot('double', name='on_referencePixelSizeDoubleSpinBox_valueChanged')
    def onReferencePixelSizeChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['curr_apix'] = value
        self.onPageChanged()
    
    @QtCore.pyqtSlot(name='on_referenceLineEdit_editingFinished)')
    def onReferenceEditChanged(self):
        '''
        '''
        
        text = str(self.ui.referenceLineEdit.text())
        self.openReference(text)
    
    @QtCore.pyqtSlot(name='on_referenceFilePushButton_clicked')
    def onReferencedOpened(self):
        '''Called when the user open reference button
        '''
        
        filename = QtGui.QFileDialog.getOpenFileName(self, self.tr("Open a reference volume"), self.lastpath)
        if filename != "": 
            self.lastpath = os.path.dirname(str(filename))
            self.ui.referenceLineEdit.blockSignals(True)
            self.ui.referenceLineEdit.setText(self.openReference(filename))
            self.ui.referenceLineEdit.blockSignals(False)
    
    def openReference(self, filename):
        ''' Open a list of micrograph files
        '''
        
        if filename == "":
            self.param['raw_reference'] = str(filename)
        if filename != "" and not os.path.exists(filename):
            QtGui.QMessageBox.warning(self, "Warning", "File does not exist: %s"%filename)

        if filename != "" and os.path.exists(filename):
            img = ndimage_file.read_image(str(filename))
            
            if img.ndim == 3:
                self.ui.referenceWidthLabel.setText(str(img.shape[0]))
                self.ui.referenceHeightLabel.setText(str(img.shape[1]))
                self.ui.referenceDepthLabel.setText(str(img.shape[2]))
                self.param['raw_reference'] = str(filename)
            else:
                QtGui.QMessageBox.warning(self, "Warning", "File is not a volume: %s"%str(img.shape))
                if self.param['curr_apix'] == 0:
                    header = ndimage_file.read_header(filename)
                    self.ui.referencePixelSizeDoubleSpinBox.blockSignals(True)
                    self.ui.referencePixelSizeDoubleSpinBox.setValue(str(header['apix']))
                    self.ui.referencePixelSizeDoubleSpinBox.blockSignals(False)
                    self.param['curr_apix'] = header['apix']
        self.onPageChanged()
        return self.param['raw_reference']
    
    ####################################################################################
    #
    # Page 1 Controls
    #
    ####################################################################################
    @QtCore.pyqtSlot(int, name='on_invertCheckBox_stateChanged')
    def onInvertContrast(self, state):
        ''' Called when the user clicks the invert contrast button
        '''
        
        if self.micrograph is not None:
            self.micrograph.invertPixels()
            self.ui.micrographDisplayLabel.setPixmap(QtGui.QPixmap.fromImage(self.micrograph))
            self.param['is_film'] = self.ui.invertCheckBox.checkState() != QtCore.Qt.Checked
    
    #@QtCore.pyqtSlot(name='on_micrographFileLineEdit_editingFinished)')
    def onMicrographLineEditChanged(self):
        '''
        '''
        
        text = str(self.ui.micrographFileLineEdit.text())
        files = []
        for filename in text.split(','):
            files.extend(glob.glob(filename))
        if len(files) > 0:
            self.openMicrograph(files)
    
    def micrographFiles(self, type='input_files'):
        ''' Get the micrograph files.
        '''
        
        text=self.param[type]
        
        files = []
        for filename in text.split(','):
            files.extend(glob.glob(filename))
        return files
    
    @QtCore.pyqtSlot(name='on_micrographFilePushButton_clicked')
    def onMicrographFileOpened(self):
        '''Called when the user clicks the Pan button.
        '''
        
        files = QtGui.QFileDialog.getOpenFileNames(self, self.tr("Open a set of micrograph images"), self.lastpath)
        if len(files) > 0:
            self.lastpath = os.path.dirname(str(files[0]))
            self.openMicrograph(files)
    
    def openMicrograph(self, files):
        ''' Open a list of micrograph files
        '''
        
        if len(files) == 0: return
        
        filenames = ','.join(compress_filenames(files))
        if filenames == self.param['orig_files']: return
        
        notfound = []
        for filename in files:
            filename = str(filename)
            if not os.path.exists(filename):
                notfound.append(filename)
        if len(notfound) > 0:
            box = QtGui.QMessageBox(QtGui.QMessageBox.Warning, "Error opening files", "Some of the files listed to not exist")
            box.setDetailedText("\n".join(notfound))
            box.exec_()
            return
        
        self.ui.micrographCountLabel.setText(str(len(files)))
        
        self.micrograph = QtGui.QImage()
        if not self.micrograph.load(files[0]):
            img = ndimage_file.read_image(str(files[0]))
            self.micrograph = numpy_to_qimage(img)
            
        self.ui.micrographWidthLabel.setText(str( self.micrograph.width()))
        self.ui.micrographHeightLabel.setText(str( self.micrograph.height()))
        
        self.param['input_files'] = filenames
        self.param['orig_files'] = filenames
        self.ui.micrographFileLineEdit.setText(filenames)
        
        if self.ui.invertCheckBox.checkState() == QtCore.Qt.Checked:
            self.micrograph.invertPixels()
        
        self.ui.micrographDisplayLabel.setPixmap(QtGui.QPixmap.fromImage(self.micrograph))
        self.onPageChanged()

def compress_filenames(files):
    ''' Test if all filenames have a similar prefix and replace the suffix
    with a wildcard ('*'). 
    
    :Parameters:
    
    files : list
            List of filenames
    
    :Returns:
    
    files : list
            List of filenames - possibly single entry with wild card
    '''
    
    files = [str(f) for f in files]
    for i in xrange(len(files)):
        if not os.path.isabs(files[i]):
            files[i] = os.path.abspath(files[i])
    
    tmp = os.path.commonprefix(files)+'*'
    if len(glob.glob(tmp)) == len(files): files = [tmp]
    return files


def _grayScaleColorModel(colortable=None):
    '''Create an RBG color table in gray scale
    
    :Parameters:
        
        colortable : list (optional)
                     Output list for QtGui.qRgb values
    :Returns:
        
        colortable : list (optional)
                     Output list for QtGui.qRgb values
    '''
    
    if colortable is None: colortable = []
    for i in xrange(256):
        colortable.append(QtGui.qRgb(i, i, i))
    return colortable
_basetable = _grayScaleColorModel()

def numpy_to_qimage(img, width=0, height=0, colortable=_basetable):
    ''' Convert a Numpy array to a PyQt4.QImage
    
    :Parameters:
    
    img : numpy.ndarray
          Array containing pixel data
    width : int
            Width of the image
    height : int
            Height of the image
    colortable : list
                 List of QtGui.qRgb values
    :Returns:
    
    qimg : PyQt4.QImage
           QImage representation
    '''
    
    if img.ndim != 2: raise ValueError, "Only gray scale images are supported for conversion, %d"%img.ndim
    
    img = ndimage_utility.normalize_min_max(img, 0, 255.0, out=img)
    img = numpy.require(img, numpy.uint8, 'C')
    h, w = img.shape
    if width == 0: width = w
    if height == 0: height = h
    qimage = QtGui.QImage(img.data, width, height, width, QtGui.QImage.Format_Indexed8)
    qimage.setColorTable(colortable)
    #qimage._numpy = img
    return qimage

        