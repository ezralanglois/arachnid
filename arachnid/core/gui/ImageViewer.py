''' Graphical user interface for displaying images

.. Created on Jul 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from pyui.MontageViewer import Ui_MainWindow
from util.qt4_loader import QtGui,QtCore,qtSlot
from util import qimage_utility
from ..metadata import spider_utility, format
from ..image import ndimage_utility, ndimage_file, ndimage_interpolate
import glob, os #, itertools
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class MainWindow(QtGui.QMainWindow):
    ''' Main window display for the plotting tool
    '''
    
    def __init__(self, parent=None):
        "Initialize a basic viewer window"
        
        QtGui.QMainWindow.__init__(self, parent)
        
        # Setup logging
        root = logging.getLogger()
        while len(root.handlers) > 0: root.removeHandler(root.handlers[0])
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(message)s'))
        root.addHandler(h)
        
        # Build window
        _logger.info("\rBuilding main window ...")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #self.connect(self.ui.zoomSlider, QtCore.SIGNAL("valueChanged(int)"), self.on_imageZoomDoubleSpinBox_valueChanged)
        
        # Setup variables
        self.lastpath = str(QtCore.QDir.currentPath())
        self.loaded_images = []
        self.files = []
        self.file_index = []
        self.color_level = None
        self.base_level = None
        
        # Image View
        self.imageListModel = QtGui.QStandardItemModel(self)
        self.ui.imageListView.setModel(self.imageListModel)
        
        # Empty init
        self.ui.actionForward.setEnabled(False)
        self.ui.actionBackward.setEnabled(False)
        self.setup()
        
    def setup(self):
        ''' Display specific setup
        '''
        
        self.ui.imageListView.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
    
    # Slots for GUI
        
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
    def on_actionLoad_More_triggered(self):
        ''' Called when someone clicks the Open Button
        '''
        
        files = glob.glob(spider_utility.spider_searchpath(self.imagefile))
        _logger.info("Found %d files on %s"%(len(files), spider_utility.spider_searchpath(self.imagefile)))
        if len(files) > 0:
            self.openImageFiles([str(f) for f in files if not format.is_readable(str(f))])
    
    @qtSlot()
    def on_actionOpen_triggered(self):
        ''' Called when someone clicks the Open Button
        '''
        
        files = QtGui.QFileDialog.getOpenFileNames(self.ui.centralwidget, self.tr("Open a set of images or documents"), self.lastpath)
        if isinstance(files, tuple): files = files[0]
        if len(files) > 0:
            self.lastpath = os.path.dirname(str(files[0]))
            self.openImageFiles([str(f) for f in files if not format.is_readable(str(f))])
    
    @qtSlot(int)
    def on_contrastSlider_valueChanged(self, value):
        ''' Called when the user uses the contrast slider
        '''
        
        if self.color_level is None: return
        if value != 200:
            self.color_level = qimage_utility.adjust_level(qimage_utility.change_contrast, self.base_level, value)
        else:
            self.color_level = self.base_level
        
        for i in xrange(len(self.loaded_images)):
            self.loaded_images[i].setColorTable(self.color_level)
            pix = QtGui.QPixmap.fromImage(self.loaded_images[i])
            icon = QtGui.QIcon(pix)
            icon.addPixmap(pix,QtGui.QIcon.Normal)
            icon.addPixmap(pix,QtGui.QIcon.Selected)
            self.imageListModel.item(i).setIcon(icon)
    
    @qtSlot(int)
    def on_zoomSlider_valueChanged(self, zoom):
        '''
        '''
        
        zoom = zoom/float(self.ui.zoomSlider.maximum())
        self.ui.imageZoomDoubleSpinBox.blockSignals(True)
        self.ui.imageZoomDoubleSpinBox.setValue(zoom)
        self.ui.imageZoomDoubleSpinBox.blockSignals(False)
        self.on_imageZoomDoubleSpinBox_valueChanged(zoom)
    
    @qtSlot(float)
    def on_imageZoomDoubleSpinBox_valueChanged(self, zoom=None):
        ''' Called when the user wants to plot only a subset of the data
        
        :Parameters:
        
        index : int
                New index in the subset combobox
        '''
        
        if zoom is None: zoom = self.ui.imageZoomDoubleSpinBox.value()
        self.ui.zoomSlider.blockSignals(True)
        self.ui.zoomSlider.setValue(int(self.ui.zoomSlider.maximum()*zoom))
        self.ui.zoomSlider.blockSignals(False)
            
        n = max(5, int(self.imagesize*zoom))
        self.ui.imageListView.setIconSize(QtCore.QSize(n, n))
    
    @qtSlot()
    def on_loadImagesPushButton_clicked(self):
        ''' Load the current batch of images into the list
        '''
        
        if len(self.files) == 0: return
        count = self.ui.imageCountSpinBox.value()
        self.imageListModel.clear()
        start = self.ui.pageSpinBox.value()*count
        index = self.file_index[start:(self.ui.pageSpinBox.value()+1)*count]
        bin_factor = self.ui.decimateSpinBox.value()
        nstd = self.ui.clampDoubleSpinBox.value()
        img = None
        self.loaded_images = []
        zoom = self.ui.imageZoomDoubleSpinBox.value()
        for i, (imgname, img) in enumerate(iter_images(self.files, index)):
            if hasattr(img, 'ndim'):
                #if not self.advanced_settings.film:
                #    ndimage_utility.invert(img, img)
                img = ndimage_utility.replace_outlier(img, nstd, nstd, replace='mean')
                if bin_factor > 1.0: img = ndimage_interpolate.interpolate(img, bin_factor)#, self.advanced_settings.downsample_type)
                qimg = qimage_utility.numpy_to_qimage(img)
            else: qimg = img
            
            if self.color_level is not None:
                qimg.setColorTable(self.color_level)
            else: 
                self.base_level = qimg.colorTable()
                self.color_level = qimage_utility.adjust_level(qimage_utility.change_contrast, self.base_level, self.ui.contrastSlider.value())
                qimg.setColorTable(self.color_level)
            self.loaded_images.append(qimg)
            pix = QtGui.QPixmap.fromImage(qimg)
            icon = QtGui.QIcon()
            icon.addPixmap(pix,QtGui.QIcon.Normal);
            icon.addPixmap(pix,QtGui.QIcon.Selected);
            item = QtGui.QStandardItem(icon, "%s/%d"%(os.path.basename(imgname[0]), imgname[1]))
            item.setData(i+start, QtCore.Qt.UserRole)
            self.imageListModel.appendRow(item)
        
        self.imagesize = img.shape[0] if hasattr(img, 'shape') else img.width()
        n = max(5, int(self.imagesize*zoom))
        self.ui.imageListView.setIconSize(QtCore.QSize(n, n))
            
        batch_count = float(len(self.file_index)/count)
        self.ui.pageSpinBox.setSuffix(" of %d"%batch_count)
        self.ui.pageSpinBox.setMaximum(batch_count)
        self.ui.actionForward.setEnabled(self.ui.pageSpinBox.value() < batch_count)
        self.ui.actionBackward.setEnabled(self.ui.pageSpinBox.value() > 0)
    
    # Other methods
    
    def openImageFiles(self, files):
        ''' Open a collection of image files, sort by content type
        
        :Parameters:
        
        files : list
                List of input files
        '''
        
        fileset=set(self.files)
        newfiles = [f for f in files if f not in fileset]
        index = len(self.files)
        self.files.extend(newfiles)
        for filename in newfiles:
            count = ndimage_file.count_images(filename)
            self.file_index.extend([(index, i) for i in xrange(count)])
            index += 1
        
        self.setWindowTitle("File count: %d - Image count: %d"%(len(self.files), len(self.file_index)))
        self.on_loadImagesPushButton_clicked()
    
def iter_images(files, index):
    ''' Wrapper for iterate images that support color PNG files
    
    :Parameters:
    
    filename : str
               Input filename
    
    :Returns:
    
    img : array
          Image array
    '''
    
    qimg = QtGui.QImage()
    if qimg.load(files[0]):
        files = [files[i[0]] for i in index]
        for filename in files:
            qimg = QtGui.QImage()
            if not qimg.load(filename): raise IOError, "Unable to read image"
            yield (filename,0), qimg
    else:
        # todo reorganize with
        '''
        for img in itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, index)):
        '''
        for f,index in index:
            img = ndimage_utility.normalize_min_max(ndimage_file.read_image(files[f], index))
            yield (files[f], index), img
        '''
        for filename in files:
            for i, img in enumerate(itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename))):
                yield (filename, i), img
        '''
            

    