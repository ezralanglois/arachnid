''' Graphical user interface for displaying images

Todo status - file being displayed

.. Created on Jul 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from pyui.MontageViewer import Ui_MainWindow
from util.qt4_loader import QtGui,QtCore,qtSlot,QtWebKit
from util import qimage_utility
import property
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
        "Initialize a image display window"
        
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
        
        # Setup variables
        self.lastpath = str(QtCore.QDir.currentPath())
        self.loaded_images = []
        self.files = []
        self.file_index = []
        self.color_level = None
        self.base_level = None
        self.inifile = '' #'ara_view.ini'
        
        # Image View
        self.imageListModel = QtGui.QStandardItemModel(self)
        self.ui.imageListView.setModel(self.imageListModel)
        
        # Empty init
        self.ui.actionForward.setEnabled(False)
        self.ui.actionBackward.setEnabled(False)
        self.setup()
        
        # Custom Actions
        
        action = self.ui.dockWidget.toggleViewAction()
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/mini/mini/application_side_list.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action.setIcon(icon8)
        self.ui.toolBar.insertAction(self.ui.actionHelp, action)
        
        # Create advanced settings
        
        property.setView(self.ui.advancedSettingsTreeView)
        self.advanced_settings, self.advanced_names = self.ui.advancedSettingsTreeView.model().addOptionList([ 
                                dict(downsample_type=('bilinear', 'ft', 'fs'), help="Choose the down sampling algorithm ranked from fastest to most accurate"),
                                dict(film=False, help="Set true to disable contrast inversion"),
                                dict(zoom=self.ui.imageZoomDoubleSpinBox.value(), help="Zoom factor where 1.0 is original size", gui=dict(readonly=True)),
                                dict(contrast=self.ui.contrastSlider.value(), help="Level of contrast in the image", gui=dict(readonly=True)),
                                dict(imageCount=self.ui.imageCountSpinBox.value(), help="Number of images to display at once", gui=dict(readonly=True)),
                                dict(decimate=self.ui.decimateSpinBox.value(), help="Number of times to reduce the size of the image in memory", gui=dict(readonly=True)),
                                dict(clamp=self.ui.clampDoubleSpinBox.value(), help="Bad pixel removal: higher the number less bad pixels removed", gui=dict(readonly=True)),
                          ])
        self.ui.advancedSettingsTreeView.setStyleSheet('QTreeView::item[readOnly="true"]{ color: #000000; }')
        
        for i in xrange(self.ui.advancedSettingsTreeView.model().rowCount()-1, 0, -1):
            if self.ui.advancedSettingsTreeView.model().index(i, 0).internalPointer().isReadOnly(): # Hide widget items (read only)
                self.ui.advancedSettingsTreeView.setRowHidden(i, QtCore.QModelIndex(), True)
        
        # Help system
        self.helpDialog = QtGui.QDialog(self)
        self.helpLayout = QtGui.QVBoxLayout(self.helpDialog)
        self.webView = QtWebKit.QWebView(self.helpDialog)
        self.helpLayout.addWidget(self.webView)
        self.helpLayout.setContentsMargins(0,0,0,0)
        
        # Load the settings
        _logger.info("\rLoading settings ...")
        self.loadSettings()
        
    def closeEvent(self, evt):
        '''Window close event triggered - save project and global settings 
        
        :Parameters:
            
        evt : QCloseEvent
              Event for to close the main window
        '''
        
        self.saveSettings()
        QtGui.QMainWindow.closeEvent(self, evt)
        
    def setup(self):
        ''' Display specific setup
        '''
        
        self.ui.imageListView.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
    
    # Slots for GUI
    @qtSlot()
    def on_actionHelp_triggered(self):
        ''' Display the help dialog
        '''
        
        self.webView.load(QtCore.QUrl('http://www.google.com'))
        self.helpDialog.show()
           
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
                if not self.advanced_settings.film:
                    ndimage_utility.invert(img, img)
                img = ndimage_utility.replace_outlier(img, nstd, nstd, replace='mean')
                if bin_factor > 1.0: img = ndimage_interpolate.interpolate(img, bin_factor, self.advanced_settings.downsample_type)
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
            self.notify_added_item(item)
        
        self.imagesize = img.shape[0] if hasattr(img, 'shape') else img.width()
        n = max(5, int(self.imagesize*zoom))
        self.ui.imageListView.setIconSize(QtCore.QSize(n, n))
            
        batch_count = float(len(self.file_index)/count)
        self.ui.pageSpinBox.setSuffix(" of %d"%batch_count)
        self.ui.pageSpinBox.setMaximum(batch_count)
        self.ui.actionForward.setEnabled(self.ui.pageSpinBox.value() < batch_count)
        self.ui.actionBackward.setEnabled(self.ui.pageSpinBox.value() > 0)
    
    # Abstract methods
    
    def notify_added_item(self, item):
        '''
        '''
        
        pass
    
    def notify_added_files(self, newfiles):
        '''
        '''
        
        pass
    
    # Other methods
    
    def openImageFiles(self, files):
        ''' Open a collection of image files, sort by content type
        
        :Parameters:
        
        files : list
                List of input files
        '''
        
        fileset=set(self.files)
        newfiles = [f for f in files if f not in fileset]
        self.notify_added_files(newfiles)
        self.updateFileIndex(newfiles)
        self.files.extend(newfiles)
        self.setWindowTitle("File count: %d - Image count: %d"%(len(self.files), len(self.file_index)))
        self.on_loadImagesPushButton_clicked()
        
    def updateFileIndex(self, newfiles):
        '''
        '''
        
        index = len(self.files)
        for filename in newfiles:
            count = ndimage_file.count_images(filename)
            self.file_index.extend([[index, i, 0] for i in xrange(count)])
            index += 1
    
    def saveSettings(self):
        ''' Save the settings of widgets
        '''
        
        if self.inifile == "": return
        settings = QtCore.QSettings(self.inifile, QtCore.QSettings.IniFormat)
        settings.setValue('main_window/geometry', self.saveGeometry())
        settings.setValue('main_window/windowState', self.saveState())
        
    def loadSettings(self):
        ''' Load the settings of widgets
        '''
        
        if self.inifile == "" or not os.path.exists(self.inifile): return
        settings = QtCore.QSettings(self.inifile, QtCore.QSettings.IniFormat)
        self.restoreGeometry(settings.value('main_window/geometry'))
        self.restoreState(settings.value('main_window/windowState'))

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
        for idx in index:
            f, i = idx[:2]
            img = ndimage_utility.normalize_min_max(ndimage_file.read_image(files[f], i))
            yield (files[f], i), img
        '''
        for filename in files:
            for i, img in enumerate(itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename))):
                yield (filename, i), img
        '''
            

    