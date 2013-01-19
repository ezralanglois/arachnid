''' Main window display for the plotting tool

.. Created on Dec 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from pyui.EmbeddingViewer import Ui_MainWindow, _fromUtf8
from PyQt4 import QtCore
from PyQt4 import QtGui

try:
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
except:
    print "Cannot import offset, upgrade matplotlib"

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.cm as cm
from matplotlib._png import read_png

from .. import format, format_utility, analysis, ndimage_file, ndimage_utility, spider_utility
import numpy, os, logging, itertools

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class MainWindow(QtGui.QMainWindow):
    ''' Main window display for the plotting tool
    '''
    
    def __init__(self, parent=None):
        "Initialize a basic viewer window"
        
        QtGui.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.lastpath = str(QtCore.QDir.currentPath())
        self.coordinates_file = ""
        self.stack_file = ""
        self.data = None
        self.label = None
        self.header = []
        self.selected = None
        self.selectedImage = None
        
        
        self.dpi = 100
        self.fig = Figure((6.0, 4.0), dpi=self.dpi)
        self.ui.canvas = FigureCanvas(self.fig)
        self.ui.canvas.setParent(self.ui.centralwidget)
        
        self.axes = self.fig.add_subplot(111)
        self.ui.mpl_toolbar = NavigationToolbar(self.ui.canvas, self)
        self.ui.mpl_toolbar.hide()
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.ui.mpl_toolbar)
        
        self.ui.centralHLayout.addWidget(self.ui.canvas)
        self.ui.centralHLayout.setStretchFactor(self.ui.canvas, 4)
        
        if not hasattr(self.ui.mpl_toolbar, 'edit_parameters'):
            self.ui.toolBar.removeAction(self.ui.actionShow_Options)
        
        self.subsetListModel = QtGui.QStandardItemModel()
        self.ui.subsetListView.setModel(self.subsetListModel)
        self.connect(self.subsetListModel, QtCore.SIGNAL("itemChanged(QStandardItem*)"), self.drawPlot)
        self.ui.imageGroupBox.setEnabled(False)
        
        action = self.ui.dockWidget.toggleViewAction()
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(_fromUtf8(":/mini/mini/application_side_list.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action.setIcon(icon8)
        self.ui.toolBar.addAction(action)
    
    @QtCore.pyqtSlot(name='on_actionPan_triggered')
    def onPanClicked(self):
        '''Called when the user clicks the Pan button.
        '''
        
        if self.ui.actionZoom.isChecked(): self.ui.actionZoom.setChecked(False)
        self.ui.mpl_toolbar.pan(self)
    
    @QtCore.pyqtSlot(name='on_actionZoom_triggered')
    def onZoomClicked(self):
        ''' Called when the user clicks the Zoom Button
        '''
        
        if self.ui.actionPan.isChecked(): self.ui.actionPan.setChecked(False)
        self.ui.mpl_toolbar.zoom( self )
    
    @QtCore.pyqtSlot(name='on_actionHome_triggered')
    def onResetView(self):
        ''' Called when the user clicks the Reset View Button
        '''
        
        if self.ui.actionZoom.isChecked(): self.ui.actionZoom.trigger()
        if self.ui.actionPan.isChecked(): self.ui.actionPan.trigger()
        self.ui.mpl_toolbar.home( self )
    
    @QtCore.pyqtSlot(name='on_actionForward_triggered')
    def onMoveForward(self):
        ''' Called when the user clicks the next view button
        '''
        
        self.ui.mpl_toolbar.forward( self )
    
    @QtCore.pyqtSlot(name='on_actionBackward_triggered')
    def onMoveBackward(self):
        ''' Called when the user clicks the previous view button
        '''
        
        self.ui.mpl_toolbar.back( self )
    
    @QtCore.pyqtSlot(name='on_actionSave_triggered')
    def onSaveFigure(self):
        ''' Called when the user clicks the Save Figure Button
        '''
        
        self.ui.mpl_toolbar.save_figure( self )
    
    @QtCore.pyqtSlot(name='on_actionShow_Options_triggered')
    def onShowOptions(self):
        ''' Called when the user clicks the Show Options Button
        '''
        
        self.ui.mpl_toolbar.edit_parameters( )
        
    @QtCore.pyqtSlot(name='on_actionOpen_triggered')
    def onOpenFile(self):
        ''' Called when someone clicks the Open Button
        '''
        
        if self.ui.actionZoom.isChecked(): self.ui.actionZoom.trigger()
        if self.ui.actionPan.isChecked(): self.ui.actionPan.trigger()
        files = QtGui.QFileDialog.getOpenFileNames(self.ui.centralwidget, self.tr("Open a set of images or documents"), self.lastpath)
        if len(files) > 0:
            self.lastpath = os.path.dirname(str(files[0]))
            self.openFiles(files)
    
    def openFiles(self, files):
        ''' Open a collection of files, sort by content type
        
        :Parameters:
        
        files : list
                List of input files
        '''
        
        for filename in files:
            filename = str(filename)
            if format.is_readable(filename):
                self.coordinates_file = filename
            else:
                self.stack_file = filename
        if self.coordinates_file != "":
            self.openFile(self.coordinates_file)
        if self.stack_file != "":
            self.ui.imageGroupBox.setEnabled(True)
            self.ui.canvas.mpl_connect('pick_event', self.displayImage)
        
    def openFile(self, filename):
        ''' Open a coordinate file
        
        :Parameters:
        
        filename : str
                   Input filename
        '''
        
        self.coordinates_file = filename
        data = format.read(self.coordinates_file, numeric=True)
        data, header = format_utility.tuple2numpy(data)
        _logger.info("Read in matrix of shape: %s"%str(data.shape))
        
        ids = numpy.ones(len(header), dtype=numpy.bool)
        try: ids[header.index('fileid')]=0
        except: pass
        try: ids[header.index('id')]=0
        except: pass
        self.label = data[:, numpy.logical_not(ids)]
        self.data = data[:, ids]
        self.selected = numpy.ones(self.data.shape[0], dtype=numpy.bool)
        self.setWindowTitle(os.path.basename(filename))
        self.header = []
        for i in xrange(len(header)):
            if ids[i]: self.header.append(header[i])
        
        try:select = self.header.index('select')
        except: select = None
        x0 = 0
        if x0 == select: x0 = 1
        y0 = x0+1 if (x0+1) < len(header) else x0
        if select is not None: select += 1

        updateComboBox(self.ui.xComboBox, self.header, index=x0)
        updateComboBox(self.ui.yComboBox, self.header, index=y0)
        updateComboBox(self.ui.colorComboBox, self.header, 'None', select)
        
        subset = []
        for i in xrange(len(self.header)):
            if numpy.alltrue(self.data[:, i]==self.data[:, i].astype(numpy.int)):
                subset.append(self.header[i])
        updateComboBox(self.ui.subsetComboBox, subset, 'None')
        
        self.clear()
        self.drawPlot()
        
    def clear(self):
        ''' Clear the plot of images
        '''
        
        clearImages(self.axes)
        self.selectedImage = None
    
    @QtCore.pyqtSlot('int', name='on_subsetComboBox_currentIndexChanged')
    def onSubsetValueChanged(self, index):
        ''' Called when the user wants to plot only a subset of the data
        
        :Parameters:
        
        index : int
                New index in the subset combobox
        '''
        
        if index > 0:
            vals = [str(v) for v in numpy.unique(self.data[:, index-1])]
        else: vals = []
        self.subsetListModel.clear()
        for name in vals:
            item = QtGui.QStandardItem(name)
            item.setCheckState(QtCore.Qt.Checked)
            item.setCheckable(True)
            self.subsetListModel.appendRow(item)
        self.drawPlot()
    
    @QtCore.pyqtSlot('int', name='on_xComboBox_currentIndexChanged')
    @QtCore.pyqtSlot('int', name='on_yComboBox_currentIndexChanged')
    @QtCore.pyqtSlot('int', name='on_colorComboBox_currentIndexChanged')
    def drawPlot(self, index=None):
        ''' Draw a scatter plot
        '''
        
        self.axes.clear()
        x = self.ui.xComboBox.currentIndex()
        y = self.ui.yComboBox.currentIndex()
        c = self.ui.colorComboBox.currentIndex()
        s = self.ui.subsetComboBox.currentIndex()
        if s > 0:
            if index is not None and hasattr(index, 'text'):
                self.clear()
                sval = float(index.text())
                if index.checkState() == QtCore.Qt.Checked:
                    self.selected = numpy.logical_or(self.selected, self.data[:, s-1]==sval)
                else:
                    self.selected = numpy.logical_and(self.selected, self.data[:, s-1]!=sval)
        data = self.data[self.selected]
        if len(data) > 0:
            color = 'r'
            cmap = None
            if c > 0:
                color = data[:, c-1]
                if color.max() != color.min():
                    color -= color.min()
                    color /= color.max()
                    cmap = cm.cool
                else: color = 'r'
            self.axes.scatter(data[:, x], data[:, y], marker='.', cmap=cmap, c=color, picker=5)
        self.ui.canvas.draw()
        self.drawImages()
        
    def displayImage(self, event):
        ''' Event invoked when user selects a data point
        
        :Parameters:
        
        event : Event
                Mouse click event
        '''
        
        if len(event.ind) > 0:
            xc = self.ui.xComboBox.currentIndex()
            yc = self.ui.yComboBox.currentIndex()
            x = event.mouseevent.xdata
            y = event.mouseevent.ydata
            ds = numpy.hypot(x-self.data[event.ind, xc], y-self.data[event.ind, yc]) # if data.shape[1] == 2 else numpy.hypot(x-data[event.ind, 0], y-data[event.ind, 1], z-data[event.ind, 2])
            self.selectedImage = event.ind[ds.argmin()]
            self.drawImages()
    
    @QtCore.pyqtSlot('int', name='on_keepSelectedCheckBox_stateChanged')
    def onKeepImageMode(self, state):
        '''
        '''

        self.clear()
        self.drawImages()
    
    @QtCore.pyqtSlot('int', name='on_imageCountSpinBox_valueChanged')
    @QtCore.pyqtSlot('int', name='on_imageSepSpinBox_valueChanged')
    @QtCore.pyqtSlot('double', name='on_imageZoomDoubleSpinBox_valueChanged')
    def drawImages(self, index=None):
        ''' Draw sample images on the plot
        '''
        
        if self.stack_file == "":
            _logger.error("No stack file - images cannot be displayed (This message indicates a bug in the code)")
            return
        total = numpy.sum(self.selected) if self.selected is not None else 0
        if total == 0: 
            _logger.info("No data plotted - images cannot be displayed")
            return
        x = self.ui.xComboBox.currentIndex()
        y = self.ui.yComboBox.currentIndex()
        zoom = self.ui.imageZoomDoubleSpinBox.value()
        radius = self.ui.imageSepSpinBox.value()
        if self.ui.keepSelectedCheckBox.checkState() == QtCore.Qt.Checked:
            if self.selectedImage is not None:
                idx = self.selectedImage
                data = self.data[self.selected]
                data = data[:, (x,y)]
                label = self.label[self.selected]
                neighbors = self.ui.averageCountSpinBox.value()
                if neighbors > 2:
                    off = idx
                    idx = numpy.argsort(numpy.hypot(data[idx, 0]-data[:, 0], data[idx, 1]-data[:, 1]))[:neighbors+1].squeeze()
                    avg = None
                    for i, img in enumerate(iter_images(self.stack_file, label[idx])):
                        if avg is None: avg = img.copy()
                        else: avg += img
                    im = OffsetImage(avg, zoom=zoom, cmap=cm.Greys_r) if img.ndim == 2 else OffsetImage(img, zoom=zoom)
                    ab = AnnotationBbox(im, data[off], xycoords='data', xybox=(radius, 0.), boxcoords="offset points", frameon=False)
                    self.axes.add_artist(ab)
                else:
                    for i, img in enumerate(iter_images(self.stack_file, label[(idx, )])):
                        im = OffsetImage(img, zoom=zoom, cmap=cm.Greys_r) if img.ndim == 2 else OffsetImage(img, zoom=zoom)
                        ab = AnnotationBbox(im, data[idx], xycoords='data', xybox=(radius, 0.), boxcoords="offset points", frameon=False)
                        self.axes.add_artist(ab)
                self.axes.plot(data[idx, 0], data[idx, 1], 'o', ms=12, alpha=0.4, color='yellow', visible=True)
        else:
            clearImages(self.axes)
            n = self.ui.imageCountSpinBox.value()
            if n == 0: 
                _logger.info("Number of images to plot is 0 - images cannot be displayed")
                self.ui.canvas.draw()
                return
            if n > total: n = total
            idx = self.selectedImage if self.selectedImage is not None else numpy.random.randint(0, self.data[self.selected].shape[0])
            if n == 1:
                data = self.data[self.selected]
                data = data[:, (x,y)]
                label = self.label[self.selected]
                for i, img in enumerate(iter_images(self.stack_file, label[(idx, )])):
                    im = OffsetImage(img, zoom=zoom, cmap=cm.Greys_r) if img.ndim == 2 else OffsetImage(img, zoom=zoom)
                    ab = AnnotationBbox(im, data[idx], xycoords='data', xybox=(radius, 0.), boxcoords="offset points", frameon=False)
                    self.axes.add_artist(ab)
                self.axes.plot(data[idx, 0], data[idx, 1], 'o', ms=12, alpha=0.4, color='yellow', visible=True)
            else:
                index = plot_random_sample_of_images(self.axes, self.stack_file, self.label, self.data[:, (x,y)], self.selected, radius, n, zoom, idx)
                self.axes.plot(self.data[index, x], self.data[index, y], 'o', ms=12, alpha=0.4, color='yellow', visible=True)
        self.ui.canvas.draw()
        
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
    
    try: read_png(filename)
    except:
        for img in itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, index)):
            yield img
    else:
        if hasattr(index, 'ndim') and index.ndim == 1:
            for i in xrange(len(index)):
                yield read_png(spider_utility.spider_filename(filename, int(index[i])))
        else:
            for img in itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, index)):
                yield img

def clearImages(ax):
    '''
    '''
    
    del ax.lines[:]
    i=0
    while i < len(ax.artists):
        obj = ax.artists[i]
        if isinstance(obj, AnnotationBbox):
            ax.artists.remove(obj)
        else: i+=1

def updateComboBox(combo, values, first=None, index=None):
    ''' Set values in a combobox without triggering a signal
    
    :Parameters:
    
    combo : QComboBox
            Target combobox
    value : list
            List of string values to add
    first : str, optional
            First value to add to combobox
    '''
    
    combo.blockSignals(True)
    combo.clear()
    if first is not None: combo.addItem(first)
    combo.addItems(values)
    if index is not None: combo.setCurrentIndex(index)
    combo.blockSignals(False)
    
def plot_random_sample_of_images(ax, stack_file, label, xy, selected, radius, n, zoom, keep=None):
    '''
    '''
    
    index = numpy.argwhere(selected)
    if index.shape[0] < 2: return
    index = index.squeeze()
    numpy.random.shuffle(index)
    off = numpy.argwhere(index==keep)
    if off.shape[0] > 0:
        off = off.squeeze()
        index[off]=index[0]
        index[0]=keep
    vals = nonoverlapping_subset(ax, xy[index], radius, n)
    index = index[vals]
    for i, img in enumerate(iter_images(stack_file, label[index])):
        im = OffsetImage(img, zoom=zoom, cmap=cm.Greys_r) if img.ndim == 2 else OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, xy[index[i]], xycoords='data', xybox=(radius, 0.), boxcoords="offset points", frameon=False)
        ax.add_artist(ab)
    return index

def nonoverlapping_subset(ax, xy, radius, n):
    ''' Find a non-overlapping subset of points on the given axes
    
    :Parameters:
    
    ax : Axes
         Current axes
    xy : array
         2 column ndarray or tuple of two ndarrays
    radius : float
             Radius of exclusion
    n : int
        Maximum number of points
    
    :Returns:
    
    index : array
            Selected indicies
    '''
    
    if isinstance(xy, tuple):
        x, y = xy
        if x.ndim == 1: x=x.reshape((x.shape[0], 1))
        if y.ndim == 1: y=y.reshape((y.shape[0], 1))
        xy = numpy.hstack((x, y))
    return analysis.subset_no_overlap(ax.transData.transform(xy), numpy.hypot(radius, radius)*2, n)

            
            