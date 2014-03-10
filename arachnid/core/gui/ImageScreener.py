''' Graphical user interface for screening images

@todo - cross index based on spider id - settings!

@todo - save

@todo - fix advanced settings load

.. Created on Jul 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ImageViewer import MainWindow as ImageViewerWindow
from util.qt4_loader import QtGui,QtCore,qtSlot
from ..metadata import format
from ..metadata import spider_utility
from ..metadata import relion_utility
from ..app import settings
from util import messagebox
import os
import numpy
import logging
import glob
#from ..util import relion_selection


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class MainWindow(ImageViewerWindow): 
    ''' Main window display for the plotting tool
    '''
    
    def __init__(self, parent=None):
        "Initialize screener window"
        
        ImageViewerWindow.__init__(self, parent)
        self.inifile = 'ara_screen.ini'
        self.settings_group = 'ImageScreener'
        self.selection_file=""
        
        # Load the settings
        _logger.info("\rLoading settings ...")
        self.loadSettings()
        
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/mini/mini/feed_disk.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ui.actionSave_Inverted = QtGui.QAction(icon8, 'Save Inverted', self)
        self.ui.actionSave_Inverted.setToolTip("Save inverted selection")
        self.ui.actionSave_Inverted.setObjectName("actionSave_Inverted")
        self.ui.toolBar.insertAction(self.ui.actionLoad_More, self.ui.actionSave_Inverted)
        #QtCore.QMetaObject.connectSlotsByName(self)
        self.ui.actionSave_Inverted.triggered.connect(self.on_actionSave_Inverted_triggered)
        
        try:
            self.selectfout = open(self.advanced_settings.select_file, 'a')
        except:  
            while True:
                
                path = QtGui.QFileDialog.getExistingDirectory(self.ui.centralwidget, self.tr("Open an existing directory to save the selections"), os.path.expanduser('~/'))
                if isinstance(path, tuple): path = path[0]
                self.inifile = os.path.join(path, 'ara_screen.ini')
                self.advanced_settings.select_file = os.path.join(path, os.path.basename(self.advanced_settings.select_file))
                if not os.path.exists(self.advanced_settings.select_file):
                    break
                else: 
                    messagebox.error_message(self, "A model file already exists in this directory! You must run ara-screen in %s"%os.path.dirname(self.advanced_settings.select_file))
                    
            self.selectfout = open(self.advanced_settings.select_file, 'a')
        self.selectedCount = 0
        self.loadSelections()
    
    def setSelectionFile(self, filename):
        '''
        '''
        
        self.selection_file=filename
    
    def showEvent(self, evt):
        '''Window close event triggered - save project and global settings 
        
        :Parameters:
            
        evt : QCloseEvent
              Event for to close the main window
        '''
        
        QtGui.QMainWindow.showEvent(self, evt)
    
    def advancedSettings(self):
        ''' Get a list of advanced settings
        '''
        
        
        return [ 
               dict(show_images=('All', 'Selected', 'Unselected'), help="Show images of specified type"),
               dict(relion="", help="Path to a relion selection file", gui=dict(filetype='open')),
               dict(select_file="ara_screen_model.csv", help="Location of output selection file", gui=dict(readonly=True)),
               dict(path_prefix=str(QtCore.QDir.currentPath()), help="Prefix for the data", gui=dict(readonly=True)),
               ]+ImageViewerWindow.sharedAdvancedSettings(self)
        
    def setup(self):
        ''' Display specific setup
        '''
        
        self.connect(self.ui.imageListView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.onSelectionChanged)
        self.ui.imageListView.setStyleSheet('QListView::item:selected{ color: #008000; border: 3px solid #6FFF00; }')
    
    def closeEvent(self, evt):
        '''Window close event triggered - save project and global settings 
        
        :Parameters:
            
        evt : QCloseEvent
              Event for to close the main window
        '''
        
        self.selectfout.close()
        ImageViewerWindow.closeEvent(self, evt)
        
    def getSettings(self):
        ''' Get the settings object
        '''
        
        '''
        return QtCore.QSettings(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, "Arachnid", "ImageScreen")
        '''
        
        if self.inifile == "": return None
        return QtCore.QSettings(self.inifile, QtCore.QSettings.IniFormat)
        
    # Loading
    def loadSelections(self):
        ''' Load the selections from the default selection file
        '''
        
        if not os.path.exists(self.advanced_settings.select_file): 
            _logger.warn("No selection file found - assuming new project (%s)"%self.advanced_settings.select_file)
            return
        
        self.files = []
        self.file_index = []
        self.selectedCount=0
        fin = open(self.advanced_settings.select_file, 'r')
        curr_path = self.advanced_settings.path_prefix
        for line in fin:
            if line =="": continue
            if line[0] == '@':
                f = line[1:].strip()
                f = os.path.join(curr_path, f)
                self.updateFileIndex([f])
                self.files.append(f)
            else:
                vals = [int(v) for v in line.strip().split(',')]
                if self.file_index[vals[0]][0] != vals[1]: raise ValueError, "Failed to load internal selection file - file id %d != %d"%(self.file_index[vals[0]][0], vals[1])
                if self.file_index[vals[0]][1] != vals[2]: raise ValueError, "Failed to load internal selection file - stack id %d != %d"%(self.file_index[vals[0]][1], vals[2])
                self.file_index[vals[0]][2]=vals[3]
                self.selectedCount += vals[3]
        fin.close()
        if len(self.files) > 0:
            self.on_loadImagesPushButton_clicked()
    
    def is_empty(self):
        '''
        '''
        
        return len(self.files) == 0
    
    # Overriden methods
    def notify_added_item(self, item):
        ''' Called when an image is added to the view
        '''
        
        if self.file_index[item.data(QtCore.Qt.UserRole), 2] > 0:
            self.ui.imageListView.selectionModel().select(self.imageListModel.indexFromItem(item), QtGui.QItemSelectionModel.Select)
    
    def notify_added_files(self, newfiles):
        ''' Called when new files are loaded
        '''
        
        for f in newfiles:
            self.selectfout.write("@%s\n"%f)
    
    def imageSubset(self, index, count):
        '''
        '''
        
        if self.advanced_settings.show_images == 'Selected':
            idx = numpy.argwhere(self.file_index[:, 2]>0)
            return self.file_index[idx[index*count:(index+1)*count].squeeze()],idx[index*count:]
        elif self.advanced_settings.show_images == 'Unselected':
            idx = numpy.argwhere(self.file_index[:, 2]==0)
            return self.file_index[idx[index*count:(index+1)*count].squeeze()],idx[index*count:]
        return ImageViewerWindow.imageSubset(self, index, count)
    
    def imageTotal(self):
        '''
        '''
        
        if self.advanced_settings.show_images == 'Selected':
            return numpy.sum(self.file_index[:, 2]>0)
        elif self.advanced_settings.show_images == 'Unselected':
            return numpy.sum(self.file_index[:, 2]==0)
        elif self.advanced_settings.show_images == 'Deleted':
            return numpy.sum(self.file_index[:, 2]<0)
        return ImageViewerWindow.imageTotal(self)
    
    # Slots for GUI
    
    #@qtSlot()
    def on_actionSave_Inverted_triggered(self):
        ''' Invert the current selection
        '''
        
        self.on_actionSave_triggered(True)
    
    @qtSlot()
    def on_actionSave_triggered(self, invert=False):
        ''' Called when someone clicks the Open Button
        '''
        
        if len(self.file_index) == 0: return
        path = self.lastpath if self.selection_file == "" else self.selection_file
        filename = QtGui.QFileDialog.getSaveFileName(self.centralWidget(), self.tr("Save selection as"), path)
        if not filename: return #selection_file
        self.saveSelection(invert, filename)
        
    def saveSelection(self, invert=False, filename=None):
        '''
        '''
        
        if filename is None: filename=self.selection_file
        self.setEnabled(False)
        progressDialog = QtGui.QProgressDialog('Saving...', "", 0,5,self)
        progressDialog.setWindowModality(QtCore.Qt.WindowModal)
        progressDialog.show()
        if isinstance(filename, tuple): filename = filename[0]
        if not filename: return
        
        file_index = self.file_index.copy()
        if invert: 
            sel = file_index[:, 2] > -1
            file_index[sel, 2] = numpy.logical_not(file_index[sel, 2]>0)
        
        if filename != "":
            if self.advanced_settings.relion != "" and os.path.splitext(filename)[1]=='.star':
                if not (len(self.files) == 1 or len(self.files) == len(file_index)):
                    progressDialog.hide()
                    QtGui.QMessageBox.critical(self, "Saving Relion Selection File", "You have opened more than one class stack. Cannot save a Relion Selection file!", QtGui.QMessageBox.Ok| QtGui.QMessageBox.Default|QtGui.QMessageBox.NoButton)
                    self.setEnabled(True)
                    return
                
                progressDialog.setValue(1)
                _logger.info("Saving Relion selection file to %s"%filename)
                class_column_name = 'rlnClassNumber'
                vals = format.read(self.advanced_settings.relion, numeric=True)
                progressDialog.setValue(2)
                subset=[]
                selected = set([v[1]+1 for v in file_index if v[2] > 0])
                progressDialog.setValue(3)
                for v in vals:
                    id = int(getattr(v, class_column_name))
                    if id in selected: subset.append(v)
                progressDialog.setValue(4)
                format.write(filename, subset)
                progressDialog.setValue(5)
                #relion_selection.select_class_subset(vals, select, filename)
            elif len(self.files) == 1 or len(self.files) == len(file_index):
                progressDialog.setValue(3)
                _logger.info("Saving single selection file to %s"%filename)
                if not spider_utility.is_spider_filename(self.files) and len(self.files) > 1:
                    _logger.info("File names do not conform to SPIDER, writing as star file")
                    filename = os.path.splitext(filename)[0]+'.star'
                    vals = [(self.files[v[0]],1) for v in file_index if v[2] > 0]
                elif len(self.files) > 1:
                    vals = [(spider_utility.spider_id(self.files[v[0]]),1) for v in file_index if v[2] > 0]
                else:
                    vals = [(v[1]+1,1) for v in file_index if v[2] > 0]
                progressDialog.setValue(4)
                format.write(filename, vals, header='id,select'.split(','), default_format=format.spidersel)
                progressDialog.setValue(5)
            else:
                progressDialog.setValue(3)
                _logger.info("Saving multiple selection files by stack to %s"%filename)
                if not spider_utility.is_spider_filename(self.files):
                    _logger.info("File names do not conform to SPIDER, writing as star file")
                    filename = os.path.splitext(filename)[0]+'.star'
                    vals = [(relion_utility.relion_identifier(self.files[v[0]], v[1]+1),1) for v in file_index if v[2] > 0]
                    format.write(filename, vals, header='id,select'.split(','))
                else:
                    micselect={}
                    for v in file_index:
                        if v[2] > 0:
                            mic = spider_utility.spider_id(self.files[v[0]])
                            if mic not in micselect: micselect[mic]=[]
                            micselect[mic].append((v[1]+1, 1))
                    for mic,vals in micselect.iteritems():
                        format.write(filename, numpy.asarray(vals), spiderid=mic, header="id,select".split(','), default_format=format.spidersel) 
                progressDialog.setValue(5)
        self.setEnabled(True)
        #progressDialog.hide()
        print 'done'
    
    @qtSlot()
    def on_loadImagesPushButton_clicked(self):
        ''' Load the current batch of images into the list
        '''
        
        self.disconnect(self.ui.imageListView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.onSelectionChanged)
        ImageViewerWindow.on_loadImagesPushButton_clicked(self)
        self.connect(self.ui.imageListView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.onSelectionChanged)
    
    # Custom Slots
    def onSelectionChanged(self, selected, deselected):
        ''' Called when the list selection has changed
        
        :Parameters:
        
        selection : QItemSelection
                    List of selection items in the list
        deselected : QItemSelection
                     List of deselected items in the list
        '''
        
        
        #modifiers = QtGui.QApplication.keyboardModifiers()
        #if modifiers == QtCore.Qt.AltModifier:
        for index in selected.indexes():
            idx = index.data(QtCore.Qt.UserRole)
            if hasattr(idx, '__iter__'): idx = idx[0]
            self.file_index[idx][2] = 1
            self.selectedCount+=1
            self.selectfout.write('%d,'%idx)
            self.selectfout.write("%d,%d,%d\n"%tuple(self.file_index[idx]))
        for index in deselected.indexes():
            idx = index.data(QtCore.Qt.UserRole)
            if hasattr(idx, '__iter__'): idx = idx[0]
            self.file_index[idx][2] = 0
            self.selectedCount-=1
            self.selectfout.write('%d,'%idx)
            self.selectfout.write("%d,%d,%d\n"%tuple(self.file_index[idx]))
        self.selectfout.flush()
        self.setWindowTitle("Selected: %d of %d"%(self.selectedCount, len(self.file_index)))
        
def launch(config_file = 'cfg/project.cfg'):
    '''
    '''
    
    
    param = settings.parse_config_simple(config_file, coordinate_file="", pow_file="", small_micrograph_file="", selection_file="") if os.path.exists(config_file) else {}
    coordinate_file = spider_utility.spider_searchpath(param.get('coordinate_file', 'local/coords/sndc000001.*'))
    small_micrograph_file = spider_utility.spider_searchpath(param.get('small_micrograph_file', 'local/mic/mic000001.*'))
    pow_file = spider_utility.spider_searchpath(param.get('pow_file', 'local/pow/pow000001.*'))
    selection_file = param.get('selection_file', 'sel_mic.dat')
    dialog = MainWindow() 
    dialog.show()
    pow_files = glob.glob(pow_file)
    mic_files = glob.glob(small_micrograph_file)
    coord_files = glob.glob(coordinate_file)
    if len(mic_files) > 0:dialog.setAlternateImage(mic_files[0], True)
    elif 'small_micrograph_file' in param: dialog.setAlternateImage(param['small_micrograph_file'], True)
    if len(coord_files) > 0: dialog.setCoordinateFile(coord_files[0], True)
    elif 'coordinate_file' in param: dialog.setCoordinateFile(param['coordinate_file'], True)
    dialog.setSelectionFile(selection_file)
    dialog.openImageFiles(pow_files)
    return dialog
