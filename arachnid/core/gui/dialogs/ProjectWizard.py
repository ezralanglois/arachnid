''' Main window display for the plotting tool

.. Created on Dec 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from pyui.ProjectWizard import Ui_ProjectWizard
from ..util import BackgroundTask
from ..util.qt4_loader import QtCore, QtGui, qtSlot, qtProperty

#from ..property import pyqtProperty
from .. import property
from .. import ndimage_file, ndimage_utility, spider_utility
#from .. import format, format_utility, analysis, ndimage_file, ndimage_utility, spider_utility
import logging, numpy, os, glob, multiprocessing, subprocess, psutil, gzip
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
        self.task=None
        self.fine_param={}
        self.leginon_filename="mapped_micrographs/mic_0000000"
        self.inifile = 'ara_project.ini'
        self.param = {'orig_files': '', 'input_files': '', 'is_film':False, 
                      'curr_apix':0.0, 'raw_reference':'',
                      'cluster_mode': project._cluster_default,
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
        self.setPixmap(QtGui.QWizard.WatermarkPixmap, QtGui.QPixmap(':/icons/icons/icon256x256.png'))
        self.setPixmap(QtGui.QWizard.BackgroundPixmap, QtGui.QPixmap(':/icons/icons/icon256x256.png'))
        #self.setPixmap(QtGui.QWizard.BannerPixmap, QtGui.QPixmap(':/icons/icons/icon64x64.png'))
        #self.setPixmap(QtGui.QWizard.LogoPixmap, QtGui.QPixmap(':/icons/icons/icon64x64.png'))
        
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
        
        #Page 2B
        self.emdbCannedModel = QtGui.QStandardItemModel(self)
        canned = [('Ribosome-70S', '2183', ':/icons/icons/ribosome_70S_32x32.png'),
                  ('Ribosome-50S', '1456', ':/icons/icons/ribosome_60S_32x32.png'),
                  ('Ribosome-30S', '5503', ':/icons/icons/ribosome30s_32x32.png'),
                  ('Ribosome-80S', '2275', ':/icons/icons/ribosome80s_32x32.png'),
                  ('Ribosome-60S', '2168', ':/icons/icons/ribosome_60S_32x32.png'),
                  ('Ribosome-40S', '1925', ':/icons/icons/ribosome_40S_32x32.png'),]
        for entry in canned:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(entry[2]), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            item = QtGui.QStandardItem(icon, entry[0])
            item.setData(entry[1], QtCore.Qt.UserRole)
            self.emdbCannedModel.appendRow(item)
        self.ui.emdbCannedListView.setModel(self.emdbCannedModel)
        self.connect(self, QtCore.SIGNAL('taskFinished(PyQt_PyObject)'), self.onDownloadFromEMDBComplete)
        
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
        
        for mode in project._cluster_modes:
            self.ui.clusterModeComboBox.addItem(mode)
        self.ui.clusterModeComboBox.setCurrentIndex(project._cluster_default)
        
        #Page 5
        
        #Page 6
        
        self.jobListModel = QtGui.QStandardItemModel(self)
        self.ui.jobListView.setModel(self.jobListModel)
        
        self.jobStatusIcons=[]
        for filename in [':/mini/mini/clock.png', ':/mini/mini/arrow_refresh.png', ':/mini/mini/tick.png', ':/mini/mini/cross.png']:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(filename), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.jobStatusIcons.append(icon)
        self.startMonitor()
        
    
    def saveSettings(self):
        ''' Save the settings of the controls in the settings map
        '''
        
        settings = QtCore.QSettings(self.inifile, QtCore.QSettings.IniFormat)
        for group, param in self.settings:
            settings.beginGroup(group)
            for name, method in param.iteritems():
                settings.setValue(name, method)
            settings.endGroup()
    
    def loadSettings(self):
        ''' Load the settings of controls specified in the settings map
        '''
        
        settings = QtCore.QSettings(self.inifile, QtCore.QSettings.IniFormat)
        for group, param in self.settings:
            settings.beginGroup(group)
            for name, val in param.iteritems():
                sval = str(settings.value(name))
                if sval is not None and sval != "" and sval != "None":
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
    # Page 7 Controls
    #
    ####################################################################################
    
    
    @qtSlot()
    def on_runJobButton_clicked(self):
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
        filename = str(self.jobListModel.item(idx).data(QtCore.Qt.UserRole))#.toString())
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
        
        try:
            id_len = spider_utility.spider_id_length(os.path.splitext(self.micrographFiles()[0])[0])
        except:
            print "*******", self.micrographFiles()
            raise
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
            
            option_list, option_groups, values = project.program.generate_settings_tree(mod, description="", **extra)
            treeView.model().addOptions(option_list, option_groups, values)
            
            #root, values = project.program.generate_settings(mod, qtProperty, QtCore.QObject, description="", **extra)
            #treeView.model().addItem(root)
            
            self.fine_param[mod]=values
            width = treeView.model().maximumTextWidth(self.fontMetrics(), treeView.indentation())
            treeView.setColumnWidth(0, width)
            treeView.expandAll()
            self.ui.treeViews.append(treeView)
    
    ####################################################################################
    #
    # Page 4 Controls
    #
    ####################################################################################  
    
    @qtSlot(int)
    def on_workerCountSpinBox_valueChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['worker_count'] = value
    
    @qtSlot(int)
    def on_threadCountSpinBox_valueChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['thread_count'] = value
    
    @qtSlot(int)
    def on_windowSizeSpinBox_valueChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['window_size'] = value
    
    @qtSlot(name='on_extensionLineEdit_editingFinished')
    def onExtensionEditChanged(self):
        '''
        '''
        
        text = str(self.ui.extensionLineEdit.text())
        self.param['ext'] = text
        self.onPageChanged()
                
    @qtSlot(int)
    def on_clusterModeComboBox_currentIndexChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['cluster_mode'] = int(value)
            
    ####################################################################################
    #
    # Page 3 Controls
    #
    ####################################################################################  
    
    @qtSlot(float)
    def on_pixelSizeDoubleSpinBox_valueChanged(self, value):
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
        
    @qtSlot(float)
    def on_voltageDoubleSpinBox_valueChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['voltage'] = value
        self.onPageChanged()
        
    @qtSlot(float)
    def on_particleSizeDoubleSpinBox_valueChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['particle_diameter'] = value
        if self.param['apix'] > 0:
            self.ui.windowSizeSpinBox.blockSignals(True)
            self.ui.windowSizeSpinBox.setValue(self.param['particle_diameter']/self.param['apix']*1.3)
            self.ui.windowSizeSpinBox.blockSignals(False)
        self.onPageChanged()
        
    @qtSlot(float)
    def on_csDoubleSpinBox_valueChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['cs'] = value
        self.onPageChanged()
    
    ####################################################################################
    #
    # Page 2 Controls
    #
    ####################################################################################   
    
    @qtSlot()
    def on_emdbDownloadPushButton_clicked(self):
        '''Called when the user clicks the download button
        '''
        
        num = self.ui.emdbNumberLineEdit.text()
        if num == "":
            QtGui.QMessageBox.warning(self, "Warning", "Empty Accession Number")
            return
        url="ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s/map/emd_%s.map.gz"%(num, num)
        self.setEnabled(False)
        self.task=BackgroundTask.launch(self, download_gunzip_task, url, '.')
    
    def onDownloadFromEMDBComplete(self, local):
        ''' Called when the download and unzip is complete
        '''
        
        self.setEnabled(True)
        self.task.disconnect()
        if local == 1:
            QtGui.QMessageBox.critical(self, "Error", "Download failed - check accession number")
        elif local == 2:
            QtGui.QMessageBox.critical(self, "Error", "Unzip failed - check file")
        else:
            self.ui.referenceLineEdit.setText(os.path.abspath(local))
            self.ui.referenceTabWidget.setCurrentIndex(0)
        self.task=None
    
    @qtSlot(QtCore.QModelIndex)
    def on_emdbCannedListView_doubleClicked(self, index):
        ''' Called when the user clicks on the list
        '''
        
        num = index.data(QtCore.Qt.UserRole) #.toString()
        self.ui.emdbNumberLineEdit.setText(num)
    
    @qtSlot(float)
    def on_referencePixelSizeDoubleSpinBox_valueChanged(self, value):
        ''' Called when the user clicks the invert contrast button
        '''
        
        self.param['curr_apix'] = value
        self.onPageChanged()
    
    @qtSlot(name='on_referenceLineEdit_editingFinished)')
    def onReferenceEditChanged(self):
        '''
        '''
        
        text = str(self.ui.referenceLineEdit.text())
        self.openReference(text)
    
    @qtSlot()
    def on_referenceFilePushButton_clicked(self):
        '''Called when the user open reference button
        '''
        
        filename = QtGui.QFileDialog.getOpenFileName(self, self.tr("Open a reference volume"), self.lastpath)
        if isinstance(filename, tuple): filename = filename[0]
        if filename != "": 
            self.lastpath = os.path.dirname(filename)
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
    @qtSlot(int)
    def on_invertCheckBox_stateChanged(self, state):
        ''' Called when the user clicks the invert contrast button
        '''
        
        if self.micrograph is not None:
            self.micrograph.invertPixels()
            self.ui.micrographDisplayLabel.setPixmap(QtGui.QPixmap.fromImage(self.micrograph))
            self.param['is_film'] = self.ui.invertCheckBox.checkState() != QtCore.Qt.Checked
    
    #@qtSlot(name='on_micrographFileLineEdit_editingFinished)')
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
        print '++++',text, type
        
        files = []
        for filename in text.split(','):
            files.extend(glob.glob(filename))
        return files
    
    @qtSlot()
    def on_micrographFilePushButton_clicked(self):
        '''Called when the user clicks the Pan button.
        '''
        
        files = QtGui.QFileDialog.getOpenFileNames(self, self.tr("Open a set of micrograph images"), self.lastpath)
        if isinstance(files, tuple): files = files[0]
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
    qimage._numpy = img
    return qimage

def download_gunzip_task(urlpath, filepath):
    ''' Download and unzip gzipped file in a separate process
    
    :Parameters:
        
    urlpath : str
              Full URL to download the file from
    filepath : str
               Local path for filename
            
    :Returns:
    
    outputfile : str
                 Output filename
    '''
    import multiprocessing
    
    def worker_callback(urlpath, filepath, qout):       
        try:
            filename=download(urlpath, filepath)
        except:
            qout.put(1)
            return
        try:
            filename=gunzip(filename)
        except:
            qout.put(2)
            return
        else:
            qout.put(filename)
    
    qout = multiprocessing.Queue()
    multiprocessing.Process(target=worker_callback, args=(urlpath, filepath, qout)).start()
    yield qout.get()

def gunzip(inputfile, outputfile=None):
    ''' Unzip a GZIPed file
    
    :Parameters:
    
    inputfile : str
                Input filename
    outputfile : str, optional
                 Output filename
                 
    :Returns:
    
    outputfile : str
                 Output filename
    '''
    
    if outputfile is None: 
        n = inputfile.rfind('.')
        outputfile=inputfile[:n]
    fin = gzip.open(inputfile, 'rb')
    fout = open(outputfile,"wb")
    fout.write(fin.read())
    fout.close()
    fin.close()
    return outputfile

def download(urlpath, filepath):
    '''Download the file at the given URL to the local filepath
    
    This function uses the urllib Python package to download file from to the remote URL
    to the local file path.
    
    :Parameters:
        
    urlpath : str
              Full URL to download the file from
    filepath : str
               Local path for filename
    
    :Returns:

    val : str
          Local filename
    '''
    import urllib
    from urlparse import urlparse
    
    filename = urllib.url2pathname(urlparse(urlpath)[2])
    filename = os.path.join(os.path.normpath(filepath), os.path.basename(filename))
    urllib.urlretrieve(urlpath, filename)
    return filename
