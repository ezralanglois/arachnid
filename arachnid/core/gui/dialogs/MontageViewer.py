''' Main window display for the plotting tool

.. Created on Dec 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from pyui.MontageViewer import Ui_MainWindow
from ..util.qt4_loader import QtGui,QtCore,qtSlot
from SettingsDialog import Dialog as SettingsDialog

from .. import ndimage_file, ndimage_utility, spider_utility, format, ndimage_interpolate #, format_utility, analysis, 
import numpy, os, logging, itertools, collections, glob, copy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class MainWindow(QtGui.QMainWindow):
    ''' Main window display for the plotting tool
    '''
    
    def __init__(self, parent=None):
        "Initialize a basic viewer window"
        
        QtGui.QMainWindow.__init__(self, parent)
        
        root = logging.getLogger()
        while len(root.handlers) > 0: root.removeHandler(root.handlers[0])
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(message)s'))
        root.addHandler(h)
        
        _logger.info("\rBuilding main window ...")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.lastpath = str(QtCore.QDir.currentPath())
        self.imagelabel = []
        self.imageids = []
        self.imagefile = ""
        self.imagesize = 0
        self.inifile = 'ara_view.ini'
        self.selectfile = 'ara_view_select.csv'
        self.selectfout = open(self.selectfile, 'a')
        self.selectedCount = 0
        self.color_level = None
        self.base_level = None
        self.image_list = []
        
        #if not os.path.exists(self.inifile): self.on_actionHelp_triggered()
        
        self.imageListModel = QtGui.QStandardItemModel(self)
        self.ui.imageListView.setModel(self.imageListModel)
        self.connect(self.ui.imageListView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.onSelectionChanged)
        self.ui.imageListView.setStyleSheet('QListView::item:selected{ color: #008000; border: 3px solid #6FFF00; }')
        
        self.fileTableModel = QtGui.QStandardItemModel(self)
        self.fileTableModel.setHorizontalHeaderLabels(['File', 'Count'])
        self.ui.fileTableView.setModel(self.fileTableModel)
        self.connect(self.ui.zoomSlider, QtCore.SIGNAL("valueChanged(int)"), self.on_imageZoomDoubleSpinBox_valueChanged)
        
        action = self.ui.dockWidget.toggleViewAction()
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/mini/mini/application_side_list.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action.setIcon(icon8)
        self.ui.toolBar.insertAction(self.ui.actionHelp, action)
        #self.ui.toolBar.addAction(action)
        self.ui.actionForward.setEnabled(False)
        self.ui.actionBackward.setEnabled(False)
        
        self.settingsDialog = SettingsDialog(self)
        self.advanced_settings, self.advanced_names = self.settingsDialog.addOptions(
                                 'Advanced', [ 
                                        dict(downsample_type=('bilinear', 'ft', 'fs'), help="Choose the down sampling algorithm ranked from fastest to most accurate"),
                                        dict(film=False, help="Set true to disable contrast inversion"),
                                        dict(zoom=self.ui.imageZoomDoubleSpinBox.value(), help="Zoom factor where 1.0 is original size", gui=dict(readonly=True)),
                                        dict(contrast=self.ui.contrastSlider.value(), help="Level of contrast in the image", gui=dict(readonly=True)),
                                        dict(imageCount=self.ui.imageCountSpinBox.value(), help="Number of images to display at once", gui=dict(readonly=True)),
                                        dict(decimate=self.ui.decimateSpinBox.value(), help="Number of times to reduce the size of the image in memory", gui=dict(readonly=True)),
                                        dict(clamp=self.ui.clampDoubleSpinBox.value(), help="Bad pixel removal: higher the number less bad pixels removed", gui=dict(readonly=True)),
                                  ])
        self.file_settings={}
                                  
        
        self.settings_map = { "main_window/geometry": (self.saveGeometry, self.restoreGeometry, None),
                              "main_window/windowState": (self.saveState, self.restoreState, None),
                              "model/files": (self.imageFiles, self.setImageFiles, None),
                              "model/imagefile": (self.imageFile, self.setImageFile, str),
                              "model/imagesize": (self.imageSize, self.setImageSize, int),
                              "model/imageids": (self.imageIDs, self.setImageIDs, None),
                             }
        """
        self.settings_map = { "main_window/geometry": (self.saveGeometry, self.restoreGeometry, 'toByteArray'),
                              "main_window/windowState": (self.saveState, self.restoreState, 'toByteArray'),
                              "model/files": (self.imageFiles, self.setImageFiles, 'toPyObject'),
                              "model/imagefile": (self.imageFile, self.setImageFile, 'toString'),
                              "model/imagesize": (self.imageSize, self.setImageSize, 'toInt'),
                              "model/imageids": (self.imageIDs, self.setImageIDs, 'toPyObject'),
                             }
        """
        for widget in vars(self.ui):
            widget = getattr(self.ui, widget)
            val = widget_settings(widget)
            if val is not None: self.settings_map.update({widget.objectName(): val})
        _logger.info("\rLoading settings ...")
        
        self.loadSettings()
        if self.imagefile != "":
            _logger.info("\rLoading selections ...")
            self.loadSelections()
            _logger.info("\rLoading images ...")
            self.on_loadImagesPushButton_clicked()
            _logger.info("\rLoading images ... done.")
            self.setWindowTitle("Selected: %d of %d"%(self.selectedCount, len(self.imagelabel)))
            self.ui.tabWidget.setCurrentIndex(0)
        else:
            self.ui.tabWidget.setCurrentIndex(1)
            files = glob.glob('local/pow/pow*.*')
            if len(files) > 0:
                tmp = glob.glob('local/mic/mic*.*')
                if len(tmp) > 0:
                    files.append(tmp[0])
                val = QtGui.QMessageBox.question(self, 'Load files?', 'Found decimated micrographs and power spectra for screening in current project directory. Would you like to load them?', QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
                if val == QtGui.QMessageBox.Yes:
                    self.openImageFiles(files)
            
         
    @qtSlot()   
    def on_actionAdvanced_Settings_triggered(self):
        ''' Display the advanced settings
        '''
        
        self.settingsDialog.exec_()
    
    @qtSlot()
    def on_actionHelp_triggered(self):
        ''' Display the help dialog
        '''
        
        box = QtGui.QMessageBox(QtGui.QMessageBox.Information, "Help", '''Welcome to Arachnid View
        
        
        This program is intended for manual selection of micrographs and power spectra simutaneously.
        
        
        See details for a short help''')
        box.setDetailedText('''Tips:
1. Open this program in the same directory each time.

2. Use the open command to add images to the TODO list. If you need to process additional data
   simply open all the micrographs again or just the additional ones.

3. Saving is automated, this program creates two files: ara_view.ini and ara_view_select.csv

4. Use the save button when finished to export selection files in SPIDER format (the extension should match your SPIDER project)

5. If you change the decimation or number of windows loaded, use the 'Reload' button to make the change

6. Click the next and back arrows at the top to go to the next group of images
        
Additional tips:

7. For speed, pre-decimate your data. If using Arachnid project
           then use the preview micrographs in local/mics/*
        
        ''')
        box.exec_()
    
    @qtSlot(int)
    def on_imageFileComboBox_currentIndexChanged(self, index):
        ''' Called when the image file combo box changes selection
        
        :Parameters:
        
        index : int
                Current index in combo box
        '''
        
        # todo save the color model
        self.storeSettings()
        self.imagefile = str(self.ui.imageFileComboBox.itemData(index))#.toString())
        self.restoreSettings()        
        self.on_loadImagesPushButton_clicked()
        
    def loadSelections(self):
        ''' Load the selections from the default selection file
        '''
        
        saved = numpy.loadtxt(self.selectfile, numpy.int, '#', ',')
        if saved.ndim == 0 or saved.shape[0]==0: saved=[]
        elif saved.ndim == 1: saved = saved.reshape((1, len(saved)))
        #if len(saved) > 0:
        #    self.updateImageFiles(numpy.unique(saved[:, 0]))
        labelmap = collections.defaultdict(dict)
        for i, val in enumerate(self.imagelabel):
            labelmap[int(val[0])][int(val[1])]=i
        _logger.info("Loaded %d micrographs from %s"%(len(saved), self.selectfile))
        for i in xrange(len(saved)):
            if i < 10:
                _logger.info("Loaded micrograph %d particle %d with selection %d -> %d"%(saved[i, 0], saved[i, 1], saved[i, 2], labelmap[saved[i, 0]][saved[i, 1]]))
            if saved[i, 0] not in labelmap:
                _logger.error("Micrograph not found: %d"%saved[i, 0])
            elif saved[i, 1] not in labelmap[saved[i, 0]]:
                _logger.error("Particle not found: %d"%saved[i, 1])
            self.imagelabel[labelmap[saved[i, 0]][saved[i, 1]], 2] = saved[i, 2]
        if len(saved) > 0:
            self.selectedCount = numpy.sum(self.imagelabel[:, 2])
     
    def onSelectionChanged(self, selected, deselected):
        ''' Called when the list selection has changed
        
        :Parameters:
        
        selection : QItemSelection
                    List of selection items in the list
        deselected : QItemSelection
                     List of deselected items in the list
        '''
        
        for index in selected.indexes():
            idx = index.data(QtCore.Qt.UserRole)
            self.imagelabel[idx, 2] = 1
            self.selectedCount+=1
            self.selectfout.write("%d,%d,%d\n"%tuple(self.imagelabel[idx, :3]))
        for index in deselected.indexes():
            idx = index.data(QtCore.Qt.UserRole)
            self.imagelabel[idx, 2] = 0
            self.selectedCount-=1
            self.selectfout.write("%d,%d,%d\n"%tuple(self.imagelabel[idx, :3]))
        self.selectfout.flush()
        self.setWindowTitle("Selected: %d of %d"%(self.selectedCount, len(self.imagelabel)))
        
    def restoreSettings(self, imagefile=None):
        ''' Restore the settings in the GUI based on the file type
        '''
        
        if imagefile is None: imagefile=self.imagefile
        if imagefile == "": return
        if imagefile not in self.file_settings:
            self.file_settings[imagefile] = copy.copy(self.advanced_settings)
        
        self.copySettings(self.file_settings[imagefile])
        setValueBlock(self.ui.zoomSlider, int(self.ui.zoomSlider.maximum()*float(self.advanced_settings.zoom)))
        setValueBlock(self.ui.contrastSlider,int(self.advanced_settings.contrast))
        setValueBlock(self.ui.imageCountSpinBox,int(self.advanced_settings.imageCount))
        setValueBlock(self.ui.decimateSpinBox,int(self.advanced_settings.decimate))
        setValueBlock(self.ui.clampDoubleSpinBox,float(self.advanced_settings.clamp))
        setValueBlock(self.ui.imageZoomDoubleSpinBox,float(self.advanced_settings.zoom))
        
    def storeSettings(self, imagefile=None):
        ''' Store the settings with associated file type
        '''
        
        if imagefile == "": return
        if imagefile is None: imagefile=self.imagefile
        self.advanced_settings.zoom = self.ui.imageZoomDoubleSpinBox.value()
        self.advanced_settings.contrast = self.ui.contrastSlider.value()
        self.advanced_settings.imageCount = self.ui.imageCountSpinBox.value()
        self.advanced_settings.decimate = self.ui.decimateSpinBox.value() 
        self.advanced_settings.clamp = self.ui.clampDoubleSpinBox.value()
        
        self.file_settings[imagefile] = copy.copy(self.advanced_settings)
            
    def copySettings(self, val):
        '''
        '''
        
        for key in self.advanced_names:
            self.advanced_settings.setProperty(key, getattr(val, key))
    
    def saveSettings(self):
        ''' Save the settings of the controls in the settings map
        '''
        
        settings = QtCore.QSettings(self.inifile, QtCore.QSettings.IniFormat)
        for key,val in self.file_settings.iteritems():
            settings.beginGroup(key)
            self.copySettings(val)
            self.settingsDialog.saveState(settings)
            settings.endGroup()
        for name, method in self.settings_map.iteritems():
            settings.setValue(name, method[0]())
    
    def loadSettings(self):
        ''' Load the settings of controls specified in the settings map
        '''
        
        if os.path.exists(self.inifile):
            settings = QtCore.QSettings(self.inifile, QtCore.QSettings.IniFormat)
            for name, method in self.settings_map.iteritems():
                try:
                    if method[2] is not None:
                        method[1](method[2](settings.value(name)))
                    else:
                        method[1](settings.value(name))
                except:
                    print '***Error', name, settings.value(name), method[2]
                    raise
            self.storeSettings()
            for key in self.imageFiles():
                settings.beginGroup(key)
                self.settingsDialog.restoreState(settings)
                self.file_settings[key] = copy.copy(self.advanced_settings)
                settings.endGroup()
            self.restoreSettings()
        
    def closeEvent(self, evt):
        '''Window close event triggered - save project and global settings 
        
        :Parameters:
            
        evt : QCloseEvent
              Event for to close the main window
        '''
        
        self.saveSettings()
        self.selectfout.close()
        QtGui.QMainWindow.closeEvent(self, evt)
    
    @qtSlot()
    def on_selectAllButton_clicked(self):
        ''' Called when the user clicks the select all button
        '''
        
        if self.selectedCount > 0:
            ret = QtGui.QMessageBox.warning(self, "Warning", "You will erase all your selections! Do you wish to continue?", QtGui.QMessageBox.Yes| QtGui.QMessageBox.No)
            if ret == QtGui.QMessageBox.No: return
        self.imagelabel[:, 2] = 1
        self.updateSelections(1)
        self.selectedCount=len(self.imagelabel)
    
    @qtSlot()
    def on_unselectAllButton_clicked(self):
        ''' Called when the user clicks the unselect all button
        '''
        
        if self.selectedCount > 0:
            ret = QtGui.QMessageBox.warning(self, "Warning", "You will erase all your selections! Do you wish to continue?", QtGui.QMessageBox.Yes| QtGui.QMessageBox.No)
            if ret == QtGui.QMessageBox.No: return
        self.imagelabel[:, 2] = 0
        self.updateSelections(0)
        self.selectedCount=0
        
    def updateSelections(self, val):
        ''' Update the visible selections and write selections to file
        '''
        
        self.disconnect(self.ui.imageListView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.onSelectionChanged)
        start = self.imageListModel.indexFromItem(self.imageListModel.item(0))
        end = self.imageListModel.indexFromItem(self.imageListModel.item(self.imageListModel.rowCount()-1))
        state = QtGui.QItemSelectionModel.Select if val > 0 else QtGui.QItemSelectionModel.Deselect
        self.ui.imageListView.selectionModel().select(QtGui.QItemSelection(start, end), state)
        self.connect(self.ui.imageListView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.onSelectionChanged)
        
        #if val == 0:
        #    self.selectfout.seek(0)
        for i in xrange(len(self.imagelabel)):
            self.selectfout.write("%d,%d,%d\n"%tuple(self.imagelabel[i, :3]))
    
    @qtSlot()
    def on_deleteModelButton_clicked(self):
        ''' Called when the user clicks the delete model button
        '''
        
        pass
        
    @qtSlot(int)
    def on_pageSpinBox_valueChanged(self, val):
        ''' Called when the user changes the group number in the spin box
        '''
        
        self.on_loadImagesPushButton_clicked()
    
    @qtSlot()
    def on_actionForward_triggered(self):
        ''' Called when the user clicks the next view button
        '''
        
        self.ui.pageSpinBox.setValue(self.ui.pageSpinBox.value()+1)
    
    @qtSlot()
    def on_actionBackward_triggered(self):
        ''' Called when the user clicks the previous view button
        '''
        
        self.ui.pageSpinBox.setValue(self.ui.pageSpinBox.value()-1)
    
    @qtSlot()
    def on_actionSave_triggered(self):
        ''' Called when the user clicks the Save Figure Button
        '''
        
        select = self.imagelabel
        filename = str(QtGui.QFileDialog.getSaveFileName(self.centralWidget(), self.tr("Save document as"), self.lastpath))
        if isinstance(filename, tuple): filename = filename[0]
        if filename != "":
            mics = numpy.unique(select[:, 0])
            if mics.shape[0] == select.shape[0]:
                select = select[:, (0,2)]
                _logger.info("Writing micrograph selection file: %s entries - %d selected"%(str(select.shape), mics.shape[0]))
                format.write(filename, select, header="id,select".split(','), format=format.spidersel)
            else:
                _logger.info("Writing particle selection files: %s entries - %d selected"%(str(select.shape), mics.shape[0]))
                for id in mics:
                    tmp = select[select[:, 0]==id, 1:]
                    tmp[:, 0]+=1
                    format.write(filename, tmp, spiderid=id, header="id,select".split(','), format=format.spidersel)
    
    @qtSlot()
    def on_actionLoad_More_triggered(self):
        ''' Called when someone clicks the Open Button
        '''
        
        files = glob.glob(spider_utility.spider_searchpath(self.imagefile))
        _logger.info("Found %d files on %s"%(len(files), spider_utility.spider_searchpath(self.imagefile)))
        if len(files) > 0:
            self.openDocumentFiles([str(f) for f in files if format.is_readable(str(f))])
            self.openImageFiles([str(f) for f in files if not format.is_readable(str(f))])
    
    @qtSlot()
    def on_actionOpen_triggered(self):
        ''' Called when someone clicks the Open Button
        '''
        
        files = QtGui.QFileDialog.getOpenFileNames(self.ui.centralwidget, self.tr("Open a set of images or documents"), self.lastpath)
        if isinstance(files, tuple): files = files[0]
        if len(files) > 0:
            self.lastpath = os.path.dirname(str(files[0]))
            self.openDocumentFiles([str(f) for f in files if format.is_readable(str(f))])
            self.openImageFiles([str(f) for f in files if not format.is_readable(str(f))])
    
    def openDocumentFiles(self, files):
        ''' Open a collection of document files, sort by content type
        
        :Parameters:
        
        files : list
                List of input files
        '''
        
        if 1 == 1:
            pass
        #DocumentModel
        '''
        if hasattr(self.imagelabel, 'ndim'): self.imagelabel = self.imagelabel.tolist()
        for id in new_ids:
            item = QtGui.QStandardItem(str(id))
            item.setToolTip(spider_utility.spider_filename(self.imagefile, id))
            count = ndimage_file.count_images(spider_utility.spider_filename(self.imagefile, id))
            item2 = QtGui.QStandardItem(str(count))
            self.fileTableModel.appendRow([item, item2])
            self.imagelabel.extend([[id, i, 0] for i in xrange(count)])
        self.imagelabel = numpy.asarray(self.imagelabel)
        '''
        #self.saveSettings()
        
        pass
    
    def openImageFiles(self, files):
        ''' Open a collection of image files, sort by content type
        
        :Parameters:
        
        files : list
                List of input files
        '''
        
        files = [str(f) for f in files]
        invalid = [filename for filename in files if not spider_utility.is_spider_filename(filename)]
        ids = [spider_utility.spider_id(filename) for filename in files if spider_utility.is_spider_filename(filename)]
        if len(ids) == 0:return
        id = ids[0] if self.imagefile == "" else self.imagefile
        imagefiles=set()
        for filename in files:
            if not spider_utility.is_spider_filename(filename): continue
            f = spider_utility.spider_filename(filename, id)
            imagefiles.add(f)
        
        _logger.info("Loading %d image files"%len(ids))
        self.ui.imageFileComboBox.blockSignals(True)
        found=None
        for imagefile in imagefiles:
            if self.ui.imageFileComboBox.findData(imagefile) == -1:
                self.ui.imageFileComboBox.addItem( os.path.basename(str(imagefile)), imagefile )
                if found is None: 
                    self.imagefile = imagefile
                    found = self.ui.imageFileComboBox.count()-1
        if found is not None: self.ui.imageFileComboBox.setCurrentIndex(found)
        self.ui.imageFileComboBox.blockSignals(False)
        '''
        if  len(files) > 0 and (self.imagefile == "" or self.imagefile != spider_utility.spider_filename(str(files[0]), self.imagefile)):
            self.ui.imageFileComboBox.blockSignals(True)
            self.ui.imageFileComboBox.addItem( os.path.basename(str(files[0])), files[0] )
            self.ui.imageFileComboBox.setCurrentIndex(self.ui.imageFileComboBox.count()-1)
            self.ui.imageFileComboBox.blockSignals(False)
            self.imagefile = files[0]
        '''
        self.updateImageFiles(ids)
        self.on_loadImagesPushButton_clicked()
        
        if len(invalid) > 0:
            box = QtGui.QMessageBox(QtGui.QMessageBox.Warning, "Warning", 'Invalid filenames skipped - do not conform to SPIDER format')
            box.setDetailedText("\n".join(invalid))
            box.exec_()
    
    def updateImageFiles(self, new_ids=None):
        ''' Update the image table and label array
        
        :Parameters:
        
        new_ids : list
                  List of new ids to add to image label array
        '''
        
        _logger.info("Updating the image label")
        if new_ids is None: new_ids = self.imageids
        else: 
            taken = set(self.imageids)
            new_ids=[id for id in new_ids if id not in taken]
            self.imageids.extend(new_ids)
        if new_ids is not None and len(new_ids) > 0:
            self.saveSettings()
            _logger.info("Added %d new image files"%len(new_ids))
        if hasattr(self.imagelabel, 'ndim'): self.imagelabel = self.imagelabel.tolist()
        for id in new_ids:
            item = QtGui.QStandardItem(str(id))
            item.setToolTip(spider_utility.spider_filename(self.imagefile, id))
            count = ndimage_file.count_images(spider_utility.spider_filename(self.imagefile, id))
            item2 = QtGui.QStandardItem(str(count))
            self.fileTableModel.appendRow([item, item2])
            self.imagelabel.extend([[id, i, 0] for i in xrange(count)])
        self.imagelabel = numpy.asarray(self.imagelabel)
    
    @qtSlot()
    def on_loadImagesPushButton_clicked(self):
        ''' Load the current batch of images into the list
        '''
        
        if len(self.imagelabel) == 0: return
        count = self.ui.imageCountSpinBox.value()
        self.disconnect(self.ui.imageListView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.onSelectionChanged)
        self.imageListModel.clear()
        start = self.ui.pageSpinBox.value()*count
        label = self.imagelabel[start:(self.ui.pageSpinBox.value()+1)*count]
        bin_factor = self.ui.decimateSpinBox.value()
        nstd = self.ui.clampDoubleSpinBox.value()
        img = None
        self.image_list = []
        zoom = self.ui.imageZoomDoubleSpinBox.value()
        for i, img in enumerate(iter_images(self.imagefile, label[:, :2])):
            if hasattr(img, 'ndim'):
                if not self.advanced_settings.film:
                    ndimage_utility.invert(img, img)
                img = ndimage_utility.replace_outlier(img, nstd, nstd, replace='mean')
                if bin_factor > 1.0: img = ndimage_interpolate.interpolate(img, bin_factor, self.advanced_settings.downsample_type)
                qimg = numpy_to_qimage(img)
            else: qimg = img
            
            if self.color_level is not None:
                qimg.setColorTable(self.color_level)
            else: 
                self.base_level = qimg.colorTable()
                self.color_level = adjust_level(change_contrast, self.base_level, self.ui.contrastSlider.value())
                qimg.setColorTable(self.color_level)
            self.image_list.append(qimg)
            pix = QtGui.QPixmap.fromImage(qimg)
            icon = QtGui.QIcon()
            icon.addPixmap(pix,QtGui.QIcon.Normal);
            icon.addPixmap(pix,QtGui.QIcon.Selected);
            item = QtGui.QStandardItem(icon, "%d/%d"%(label[i, 0], label[i, 1]+1))
            item.setData(i+start, QtCore.Qt.UserRole)
            self.imageListModel.appendRow(item)
            #_logger.info("%d -> %s"%(i, str(label[i, :3])))
            if label[i, 2] > 0:
                self.ui.imageListView.selectionModel().select(self.imageListModel.indexFromItem(item), QtGui.QItemSelectionModel.Select)
        self.connect(self.ui.imageListView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.onSelectionChanged)
        #if self.imagesize == 0:
        #    self.imagesize = img.shape[0] if hasattr(img, 'shape') else img.width()
        #    self.on_imageZoomDoubleSpinBox_valueChanged()
        
        self.imagesize = img.shape[0] if hasattr(img, 'shape') else img.width()
        n = max(5, int(self.imagesize*zoom))
        self.ui.imageListView.setIconSize(QtCore.QSize(n, n))
            
        batch_count = float(len(self.imagelabel)/count)
        self.ui.pageSpinBox.setSuffix(" of %d"%batch_count)
        self.ui.pageSpinBox.setMaximum(batch_count)
        self.ui.actionForward.setEnabled(self.ui.pageSpinBox.value() < batch_count)
        self.ui.actionBackward.setEnabled(self.ui.pageSpinBox.value() > 0)
    
    @qtSlot(int)
    def on_contrastSlider_valueChanged(self, value):
        ''' Called when the user uses the contrast slider
        '''
        
        if self.color_level is None: return
        if value != 200:
            self.color_level = adjust_level(change_contrast, self.base_level, value)
        else:
            self.color_level = self.base_level
        
        for i in xrange(len(self.image_list)):
            self.image_list[i].setColorTable(self.color_level)
            pix = QtGui.QPixmap.fromImage(self.image_list[i])
            icon = QtGui.QIcon(pix)
            icon.addPixmap(pix,QtGui.QIcon.Normal)
            icon.addPixmap(pix,QtGui.QIcon.Selected)
            self.imageListModel.item(i).setIcon(icon)
    
    @qtSlot(float)
    def on_imageZoomDoubleSpinBox_valueChanged(self, zoom=None):
        ''' Called when the user wants to plot only a subset of the data
        
        :Parameters:
        
        index : int
                New index in the subset combobox
        '''
        
        if zoom is None: zoom = self.ui.imageZoomDoubleSpinBox.value()
        elif isinstance(zoom, int): 
            zoom = zoom/float(self.ui.zoomSlider.maximum())
            self.ui.imageZoomDoubleSpinBox.blockSignals(True)
            self.ui.imageZoomDoubleSpinBox.setValue(zoom)
            self.ui.imageZoomDoubleSpinBox.blockSignals(False)
        else:
            self.ui.zoomSlider.blockSignals(True)
            self.ui.zoomSlider.setValue(int(self.ui.zoomSlider.maximum()*zoom))
            self.ui.zoomSlider.blockSignals(False)
            
        n = max(5, int(self.imagesize*zoom))
        self.ui.imageListView.setIconSize(QtCore.QSize(n, n))
    
    def setImageFiles(self, files):
        ''' Set list of image files
        
        :Parameters:
        
        files : list
                List of image filenames
        '''
        
        if files is not None:
            self.ui.imageFileComboBox.blockSignals(True)
            for filename in files:
                self.ui.imageFileComboBox.addItem( os.path.basename(str(filename)), filename )
            self.ui.imageFileComboBox.blockSignals(False)
    
    def imageFiles(self):
        ''' Get list of image files
        
        :Returns:
        
        files : list
                Image filenames
        '''
        
        files = []
        for i in xrange(self.ui.imageFileComboBox.count()):
            files.append(str(self.ui.imageFileComboBox.itemData(i)))#.toString()))
        return files
        
    def imageFile(self):
        ''' Get the filename for the images
        
        :Returns:
        
        filename : str
                   Image filename
        '''
        
        return self.imagefile
    
    def setImageFile(self, f):
        ''' Set the filename for the images
        
        :Parameters:
        
        f : str
            New filename for image
        '''
        
        f = str(f)
        if f != "" and not os.path.exists(f):
            _logger.info("File does not exist: %s"%f)
            self.lastpath = os.path.dirname(f)
            while True:
                f = QtGui.QFileDialog.getOpenFileName(self.ui.centralwidget, self.tr("Open an image - %s"%os.path.basename(f)), self.lastpath)
                f = str(f[0]) if isinstance(f, tuple) else str(f)
                if f == "":
                    ret = QtGui.QMessageBox.warning(self, "Warning", "The image file you are trying to use does not exist, do you wish to exit?", QtGui.QMessageBox.Yes| QtGui.QMessageBox.No)
                    if ret == QtGui.QMessageBox.Yes: self.close()
                else: break
        self.imagefile = f
        if self.ui.imageFileComboBox.count() == 0:
            if f is not None and f != "":
                self.ui.imageFileComboBox.blockSignals(True)
                self.ui.imageFileComboBox.addItem( os.path.basename(str(f)), f )
                self.ui.imageFileComboBox.blockSignals(False)
        else:
            index = self.ui.imageFileComboBox.findText(os.path.basename(str(f)))
            self.ui.imageFileComboBox.blockSignals(True)
            self.ui.imageFileComboBox.setCurrentIndex(index)
            self.ui.imageFileComboBox.blockSignals(False)
    
    def imageSize(self):
        ''' Get the size of an image
        
        :Returns:
        
        size : int
               Size of the image
        '''
        
        return self.imagesize
    
    def setImageSize(self, val):
        ''' Set the size of an image
        
        :Parameters:
        
        val : int
               Size of the image
        '''
        
        if val is None or val == "": val = 0
        val = int(val)
        self.imagesize = val
        
    def imageIDs(self):
        ''' Get the label array describing all images
        
        :Returns:
        
        label : array
                Label array
        '''
        
        return self.imageids
    
    def setImageIDs(self, label):
        ''' Set the label array for the images
        
        :Parameters:
        
        label : int
                Label array
        '''
        
        if label is None: label = []
        self.imageids = []
        saved=set()
        for id in label:
            id = int(id)
            if id not in saved:
                saved.add(id)
                self.imageids.append(id)
        
        self.updateImageFiles()


        
def setValueBlock(widget, val):
    ''' Set the value of a widget while blocking events
    
    :Parameters:
    
    widget : QWidget
             Widget to set value
    val : object
          Value to set
    '''
    
    widget.blockSignals(True)
    widget.setValue(val)
    widget.blockSignals(False)
    
def iter_images(filename, index):
    ''' Wrapper for iterate images that support color PNG files
    
    :Parameters:
    
    filename : str
               Input filename
    index : array
            List of file and image Ids
    
    :Returns:
    
    img : array
          Image array
    '''
    
    qimg = QtGui.QImage()
    if qimg.load(filename):
        index = numpy.asarray(index)
        if index.ndim == 2: index = index[:, 0]
        for i in xrange(len(index)):
            if not qimg.load(spider_utility.spider_filename(filename, int(index[i]))): raise IOError, "Unable to read image"
            yield qimg
    else:
        for img in itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, index)):
            yield img

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
    #img = histeq(img)
    img = ndimage_utility.normalize_min_max(img, 0, 255.0, out=img)
    img = numpy.require(img, numpy.uint8, 'C')
    h, w = img.shape
    if width == 0: width = w
    if height == 0: height = h
    qimage = QtGui.QImage(img.data, width, height, width, QtGui.QImage.Format_Indexed8)
    qimage.setColorTable(colortable)
    qimage._numpy = img
    return qimage

"""
def widget_type(val):
    '''
    '''
    
    if isinstance(val, float):
        return float
    elif isinstance(val, int):
        return int
    else: return ValueError, "Cannot find type for: %s -- %s"%(str(val), val.__class__.__name__)
"""

def widget_settings(widget):
     ''' Generate settings callbacks for various widget types
     
     :Parameters:
     
     widget : QWidget
              Widget state to maintain
    
    :Returns:
    
    getter : function
             Get value of the widget
    setter : function
             Set value of the function
    type : str
           Name of function to convert value
     '''
     
     if hasattr(widget, 'setValue') and hasattr(widget, 'value'):
         return (widget.value, widget.setValue, widget.value().__class__)
     return None
 
def histeq(img, hist_bins=256, **extra):
    ''' Equalize the histogram of an image
    
    :Parameters:
    
    img : array
          Image data
    hist_bins : int
                Number of bins for the histogram
    
    :Returns:
    
    img : array
          Histogram equalized image
          
    .. note::
    
        http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    '''
    
    imhist,bins = numpy.histogram(img.flatten(),hist_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    im2 = numpy.interp(img.flatten(), bins[:-1], cdf)#use linear interpolation of cdf to find new pixel values
    img = im2.reshape(img.shape)
    return img

def change_brightness(value, brightness):
    ''' Change pixel brightness by some factor
    '''
    
    return min(max(value + brightness * 255 / 100, 0), 255)

def change_contrast(value, contrast):
    ''' Change pixel contrast by some factor
    '''
    
    return min(max(((value-127) * contrast / 100) + 127, 0), 255)

def change_gamma(value, gamma):
    ''' Change pixel gamma by some factor
    '''
    
    return min(max(int( numpy.pow( value/255.0, 100.0/gamma ) * 255 ), 0), 255)

def adjust_level(func, colorTable, level):
    ''' Adjust the color level of an image
    
    :Parameters:
    
    func : function
           Adjustment function
    colorTable : list
                 List of QtGui.qRgb values
    level : int
            Current color level (0 - 255)
    
    :Returns:
    
    colorTable : list
                 List of QtGui.qRgb values
    '''
    
    table = []
    for color in colorTable:
        r = func(QtGui.qRed(color), level)
        g = func(QtGui.qGreen(color), level)
        b = func(QtGui.qBlue(color), level)
        table.append(QtGui.qRgb(r, g, b))
    return table
    
 
def customEmit(self, record):
    # Monkey patch Emit function to avoid new lines between records
    import types
    try:
        msg = self.format(record)
        if not hasattr(types, "UnicodeType"): #if no unicode support...
            self.stream.write(msg)
        else:
            try:
                if getattr(self.stream, 'encoding', None) is not None:
                    self.stream.write(msg.encode(self.stream.encoding))
                else:
                    self.stream.write(msg)
            except UnicodeError:
                self.stream.write(msg.encode("UTF-8"))
        self.flush()
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        self.handleError(record)
            
#setattr(logging.StreamHandler, logging.StreamHandler.emit.__name__, customEmit)

