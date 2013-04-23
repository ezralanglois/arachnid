''' Main window display for the plotting tool

.. Created on Dec 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from pyui.MontageViewer import Ui_MainWindow, _fromUtf8
from PyQt4 import QtCore
from PyQt4 import QtGui

from .. import ndimage_file, ndimage_utility, spider_utility, eman2_utility, format #, format_utility, analysis, 
import numpy, os, logging, itertools, collections

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
        
        if not os.path.exists(self.inifile): self.displayHelp()
        
        self.imageListModel = QtGui.QStandardItemModel(self)
        self.ui.imageListView.setModel(self.imageListModel)
        self.connect(self.ui.imageListView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.onSelectionChanged)
        self.ui.imageListView.setStyleSheet('QListView::item:selected{ color: #008000; border: 3px solid #6FFF00; }')
        
        self.fileTableModel = QtGui.QStandardItemModel(self)
        self.fileTableModel.setHorizontalHeaderLabels(['File', 'Count'])
        self.ui.fileTableView.setModel(self.fileTableModel)
        self.connect(self.ui.zoomSlider, QtCore.SIGNAL("valueChanged(int)"), self.onZoomValueChanged)
        
        action = self.ui.dockWidget.toggleViewAction()
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(_fromUtf8(":/mini/mini/application_side_list.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action.setIcon(icon8)
        self.ui.toolBar.insertAction(self.ui.actionHelp, action)
        #self.ui.toolBar.addAction(action)
        self.ui.actionForward.setEnabled(False)
        self.ui.actionBackward.setEnabled(False)
        
        self.settings_map = { "main_window/geometry": (self.saveGeometry, self.restoreGeometry, 'toByteArray'),
                              "main_window/windowState": (self.saveState, self.restoreState, 'toByteArray'),
                              "model/files": (self.imageFiles, self.setImageFiles, 'toPyObject'),
                              "model/imagefile": (self.imageFile, self.setImageFile, 'toString'),
                              "model/imagesize": (self.imageSize, self.setImageSize, 'toInt'),
                              "model/imageids": (self.imageIDs, self.setImageIDs, 'toPyObject'),
                             }
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
            self.loadImages()
            _logger.info("\rLoading images ... done.")
            self.setWindowTitle("Selected: %d of %d"%(self.selectedCount, len(self.imagelabel)))
    
    @QtCore.pyqtSlot(name='on_actionHelp_triggered')
    def displayHelp(self):
        ''' Display the help dialog
        '''
        
        box = QtGui.QMessageBox(QtGui.QMessageBox.Information, "Help", '''Tips:
1. Open this program in the same directory each time.

2. Saving is automated, this program creates two files: ara_view.ini and ara_view_select.csv

3. Use the save button when finished to export selection files in SPIDER format

4. If you change the decimation or number of windows loaded, use the 'Reload' button to make the change

5. Click the next and back arrows at the top to go to the next group of images
            ''')
        #box.setDetailedText(
        box.exec_()
    
    @QtCore.pyqtSlot('int', name='on_imageFileComboBox_currentIndexChanged')
    def onImageFileChanged(self, index):
        ''' Called when the image file combo box changes selection
        
        :Parameters:
        
        index : int
                Current index in combo box
        '''
        
        self.imagefile = str(self.ui.imageFileComboBox.itemData(index).toString())
        self.loadImages()
        
    def loadSelections(self):
        ''' Load the selections from the default selection file
        '''
        
        labelmap = collections.defaultdict(dict)
        for i, val in enumerate(self.imagelabel):
            labelmap[int(val[0])][int(val[1])]=i
        saved = numpy.loadtxt(self.selectfile, numpy.int, '#', ',')
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
            idx = index.data(QtCore.Qt.UserRole).toPyObject()
            self.imagelabel[idx, 2] = 1
            self.selectedCount+=1
            self.selectfout.write("%d,%d,%d\n"%tuple(self.imagelabel[idx, :3]))
        for index in deselected.indexes():
            idx = index.data(QtCore.Qt.UserRole).toPyObject()
            self.imagelabel[idx, 2] = 0
            self.selectedCount-=1
            self.selectfout.write("%d,%d,%d\n"%tuple(self.imagelabel[idx, :3]))
        self.selectfout.flush()
        self.setWindowTitle("Selected: %d of %d"%(self.selectedCount, len(self.imagelabel)))
    
    def saveSettings(self):
        ''' Save the settings of the controls in the settings map
        '''
        
        settings = QtCore.QSettings(self.inifile, QtCore.QSettings.IniFormat)
        for name, method in self.settings_map.iteritems():
            settings.setValue(name, QtCore.QVariant(method[0]()))
    
    def loadSettings(self):
        ''' Load the settings of controls specified in the settings map
        '''
        
        if os.path.exists(self.inifile):
            settings = QtCore.QSettings(self.inifile, QtCore.QSettings.IniFormat)
            for name, method in self.settings_map.iteritems():
                val = getattr(settings.value(name), method[2])()
                if isinstance(val, tuple): val = val[0]
                method[1](val)
        
    def closeEvent(self, evt):
        '''Window close event triggered - save project and global settings 
        
        :Parameters:
            
        evt : QCloseEvent
              Event for to close the main window
        '''
        
        self.saveSettings()
        self.selectfout.close()
        QtGui.QMainWindow.closeEvent(self, evt)
    
    @QtCore.pyqtSlot(name='on_selectAllButton_clicked')
    def onSelectAll(self):
        ''' Called when the user clicks the select all button
        '''
        
        if self.selectedCount > 0:
            ret = QtGui.QMessageBox.warning(self, "Warning", "You will erase all your selections! Do you wish to continue?", QtGui.QMessageBox.Yes| QtGui.QMessageBox.No)
            if ret == QtGui.QMessageBox.No: return
        self.imagelabel[:, 2] = 1
        self.updateSelections(1)
        self.selectedCount=len(self.imagelabel)
    
    @QtCore.pyqtSlot(name='on_unselectAllButton_clicked')
    def onUnselectAll(self):
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
    
    @QtCore.pyqtSlot(name='on_deleteModelButton_clicked')
    def onDeleteModel(self):
        ''' Called when the user clicks the delete model button
        '''
        
        pass
        
    @QtCore.pyqtSlot('int', name='on_pageSpinBox_valueChanged')
    def onMoveGroup(self, val):
        ''' Called when the user changes the group number in the spin box
        '''
        
        self.loadImages()
    
    @QtCore.pyqtSlot(name='on_actionForward_triggered')
    def onMoveForward(self):
        ''' Called when the user clicks the next view button
        '''
        
        self.ui.pageSpinBox.setValue(self.ui.pageSpinBox.value()+1)
    
    @QtCore.pyqtSlot(name='on_actionBackward_triggered')
    def onMoveBackward(self):
        ''' Called when the user clicks the previous view button
        '''
        
        self.ui.pageSpinBox.setValue(self.ui.pageSpinBox.value()-1)
    
    @QtCore.pyqtSlot(name='on_actionSave_triggered')
    def onSaveSelections(self):
        ''' Called when the user clicks the Save Figure Button
        '''
        
        select = self.imagelabel
        filename = str(QtGui.QFileDialog.getSaveFileName(self.centralWidget(), self.tr("Save document as"), self.lastpath))
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
        
    @QtCore.pyqtSlot(name='on_actionOpen_triggered')
    def onOpenFile(self):
        ''' Called when someone clicks the Open Button
        '''
        
        files = QtGui.QFileDialog.getOpenFileNames(self.ui.centralwidget, self.tr("Open a set of images or documents"), self.lastpath)
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
        if (self.imagefile == "" or self.imagefile != spider_utility.spider_filename(str(files[0]), self.imagefile)) and len(files) > 0:
            self.ui.imageFileComboBox.blockSignals(True)
            self.ui.imageFileComboBox.addItem( os.path.basename(str(files[0])), files[0] )
            self.ui.imageFileComboBox.blockSignals(False)
            self.imagefile = files[0]
        taken = set(self.imageids)
        self.updateImageFiles([id for id in ids if id not in taken])
        self.loadImages()
        
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
        else: self.imageids.extend(new_ids)
        if hasattr(self.imagelabel, 'ndim'): self.imagelabel = self.imagelabel.tolist()
        for id in new_ids:
            item = QtGui.QStandardItem(str(id))
            item.setToolTip(spider_utility.spider_filename(self.imagefile, id))
            count = ndimage_file.count_images(spider_utility.spider_filename(self.imagefile, id))
            item2 = QtGui.QStandardItem(str(count))
            self.fileTableModel.appendRow([item, item2])
            self.imagelabel.extend([[id, i, 0] for i in xrange(count)])
        self.imagelabel = numpy.asarray(self.imagelabel)
    
    @QtCore.pyqtSlot(name='on_loadImagesPushButton_clicked')
    def loadImages(self):
        ''' Load the current batch of images into the list
        '''
        
        if len(self.imagelabel) == 0: return
        count = self.ui.imageCountSpinBox.value()
        self.imageListModel.clear()
        start = self.ui.pageSpinBox.value()*count
        label = self.imagelabel[start:(self.ui.pageSpinBox.value()+1)*count]
        bin_factor = self.ui.decimateSpinBox.value()
        nstd = self.ui.clampDoubleSpinBox.value()
        img = None
        self.disconnect(self.ui.imageListView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.onSelectionChanged)
        self.image_list = []
        for i, img in enumerate(iter_images(self.imagefile, label[:, :2])):
            if hasattr(img, 'ndim'):
                img = ndimage_utility.replace_outlier(img, nstd, nstd, replace='mean')
                if bin_factor > 1.0: img = eman2_utility.decimate(img, bin_factor)
                qimg = numpy_to_qimage(img)
            else: qimg = img
            
            if self.color_level is not None:
                qimg.setColorTable(self.color_level)
            else: 
                self.base_level = qimg.colorTable()
                self.color_level = adjust_level(change_contrast, self.base_level, self.ui.contrastSlider.value())
                qimg.setColorTable(self.color_level)
            self.image_list.append(QtGui.QImage(qimg))
            pix = QtGui.QPixmap.fromImage(qimg)
            icon = QtGui.QIcon()
            icon.addPixmap(pix,QtGui.QIcon.Normal);
            icon.addPixmap(pix,QtGui.QIcon.Selected);
            item = QtGui.QStandardItem(icon, "%d/%d"%(label[i, 0], label[i, 1]+1))
            item.setData(i+start, QtCore.Qt.UserRole)
            self.imageListModel.appendRow(item)
            _logger.info("%d -> %s"%(i, str(label[i, :3])))
            if label[i, 2] > 0:
                self.ui.imageListView.selectionModel().select(self.imageListModel.indexFromItem(item), QtGui.QItemSelectionModel.Select)
        self.connect(self.ui.imageListView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.onSelectionChanged)
        if self.imagesize == 0:
            self.imagesize = img.shape[0] if hasattr(img, 'shape') else img.width()
            self.onZoomValueChanged()
            
        batch_count = float(len(self.imagelabel)/count)
        self.ui.pageSpinBox.setSuffix(" of %d"%batch_count)
        self.ui.pageSpinBox.setMaximum(batch_count)
        self.ui.actionForward.setEnabled(self.ui.pageSpinBox.value() < batch_count)
        self.ui.actionBackward.setEnabled(self.ui.pageSpinBox.value() > 0)
    
    @QtCore.pyqtSlot('int', name='on_contrastSlider_valueChanged')
    def onContrastChanged(self, value):
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
        
    #@QtCore.pyqtSlot('double', name='on_zoomSlider_valueChanged')
    @QtCore.pyqtSlot('double', name='on_imageZoomDoubleSpinBox_valueChanged')
    def onZoomValueChanged(self, zoom=None):
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
            files.append(str(self.ui.imageFileComboBox.itemData(i).toString()))
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
                f = str(QtGui.QFileDialog.getOpenFileName(self.ui.centralwidget, self.tr("Open an image - %s"%os.path.basename(f)), self.lastpath))
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
    #qimage._numpy = img
    return qimage

def widget_type(val):
    '''
    '''
    
    if isinstance(val, float):
        return 'toFloat'
    elif isinstance(val, int):
        return 'toInt'
    else: return ValueError, "Cannot find type for: %s -- %s"%(str(val), val.__class__.__name__)

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
         return (widget.value, widget.setValue, widget_type(widget.value()))
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

