''' Main window display for the plotting tool

.. Created on Dec 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from pyui.EmbeddingViewer import Ui_MainWindow
from ..util.qt4_loader import QtCore, QtGui, qtSlot

if not hasattr(QtCore, 'pyqtSlot'):
    import matplotlib
    matplotlib.rcParams['backend.qt4']='PySide'

try:
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
except:
    print "Cannot import offset, upgrade matplotlib"

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.cm as cm, matplotlib.lines
from matplotlib._png import read_png

from .. import format, analysis, ndimage_file, ndimage_utility, spider_utility, rotate
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
        self.rtsq=None
        self.groups=None
        self.highlight_index = 0
        default_markers=['s', 'o', '^', '>', 'v', 'd', 'p', 'h', '8', '+', 'x']
        self.markers=list(default_markers)
        if 1 == 0:
            default_markers=set(default_markers)
    
            
            for m in matplotlib.lines.Line2D.markers:
                try:
                    if len(m) == 1 and m != ' ' and m not in default_markers:
                        self.markers.append(m)
                except TypeError:
                    pass
        
        self.fig = Figure((6.0, 4.0), dpi=self.ui.dpiSpinBox.value()) #set_dpi
        self.ui.canvas = FigureCanvas(self.fig)
        self.ui.canvas.setParent(self.ui.centralwidget)
        #self.connect(self.ui.dpiSpinBox, QtCore.SIGNAL('valueChanged(int)'), self.fig.set_dpi)
        
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
        #self.connect(self.subsetListModel, QtCore.SIGNAL("itemChanged(QStandardItem*)"), self.drawPlot)
        self.ui.imageGroupBox.setEnabled(False)
        
        action = self.ui.dockWidget.toggleViewAction()
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/mini/mini/application_side_list.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action.setIcon(icon8)
        self.ui.toolBar.addAction(action)
    
    
    @qtSlot()
    def on_selectAllPushButton_clicked(self):
        '''
        '''
        
        _logger.debug("select")
        for i in xrange(self.subsetListModel.rowCount()):
            self.subsetListModel.item(i).setCheckState(QtCore.Qt.Checked)
    
    @qtSlot()
    def on_unselectAllPushButton_clicked(self):
        '''
        '''
        
        _logger.debug("unselect")
        for i in xrange(self.subsetListModel.rowCount()):
            self.subsetListModel.item(i).setCheckState(QtCore.Qt.Unchecked)
        
    @qtSlot(int)
    def on_dpiSpinBox_valueChanged(self, val):
        '''
        '''
        
        dpi = self.fig.get_dpi()
        w = self.fig.get_figwidth()
        h = self.fig.get_figheight()
        self.fig.set_size_inches(w*val/dpi, h*val/dpi)
        print w, h, w*val/dpi, h*val/dpi
        self.fig.set_dpi(val)
    
    @qtSlot()
    def on_actionPan_triggered(self):
        '''Called when the user clicks the Pan button.
        '''
        
        if self.ui.actionZoom.isChecked(): self.ui.actionZoom.setChecked(False)
        self.ui.mpl_toolbar.pan(self)
    
    @qtSlot()
    def on_actionZoom_triggered(self):
        ''' Called when the user clicks the Zoom Button
        '''
        
        if self.ui.actionPan.isChecked(): self.ui.actionPan.setChecked(False)
        self.ui.mpl_toolbar.zoom( self )
    
    @qtSlot()
    def on_actionHome_triggered(self):
        ''' Called when the user clicks the Reset View Button
        '''
        
        if self.ui.actionZoom.isChecked(): self.ui.actionZoom.trigger()
        if self.ui.actionPan.isChecked(): self.ui.actionPan.trigger()
        self.ui.mpl_toolbar.home( self )
    
    @qtSlot()
    def on_actionForward_triggered(self):
        ''' Called when the user clicks the next view button
        '''
        
        self.ui.mpl_toolbar.forward( self )
    
    @qtSlot()
    def on_actionBackward_triggered(self):
        ''' Called when the user clicks the previous view button
        '''
        
        self.ui.mpl_toolbar.back( self )
    
    @qtSlot()
    def on_actionSave_triggered(self):
        ''' Called when the user clicks the Save Figure Button
        '''
        
        self.ui.mpl_toolbar.save_figure( self )
    
    @qtSlot()
    def on_actionShow_Options_triggered(self):
        ''' Called when the user clicks the Show Options Button
        '''
        
        self.ui.mpl_toolbar.edit_parameters( )
        
    @qtSlot()
    def on_actionOpen_triggered(self):
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
            if not ndimage_file.spider.is_readable(filename) and not ndimage_file.mrc.is_readable(filename) and  format.is_readable(filename):
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
        data, header = format.read(self.coordinates_file, ndarray=True)
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
        self.headermap={}
        for i in xrange(len(header)):
            if ids[i]: 
                self.headermap[header[i]]=len(self.header)
                self.header.append(header[i])
        try:
            self.rtsq = (self.header.index('psi'), self.header.index('tx'), self.header.index('ty'))
        except:
            self.rtsq = None
        
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
        updateComboBox(self.ui.selectGroupComboBox, subset, 'None')
        self.ui.selectGroupComboBox.setEnabled(False)
        
        self.clear()
        self.drawPlot()
        
    def clear(self):
        ''' Clear the plot of images
        '''
        
        clearImages(self.axes)
        self.selectedImage = None
    
    @qtSlot(int)
    def on_selectGroupComboBox_currentIndexChanged(self, index):
        ''' Called when the user wants to highlight only a subset of the data
        
        :Parameters:
        
        index : int
                New index in the subset combobox
        '''
        
        self.highlight_index = int(index)
        
    @qtSlot(int)
    def on_subsetComboBox_currentIndexChanged(self, index):
        ''' Called when the user wants to plot only a subset of the data
        
        :Parameters:
        
        index : int
                New index in the subset combobox
        '''
        
        if index > 0:
            index = self.headermap[str(self.ui.subsetComboBox.currentText())]
            print index, self.ui.subsetComboBox.currentText()
            vals = [str(v) for v in numpy.unique(self.data[:, index])]
        else: vals = []
        self.subsetListModel.clear()
        for name in vals:
            item = QtGui.QStandardItem(name)
            item.setCheckState(QtCore.Qt.Checked)
            item.setCheckable(True)
            self.subsetListModel.appendRow(item)
        #self.drawPlot()
    
    
    #@qtSlot('int', name='on_xComboBox_currentIndexChanged')
    #@qtSlot('int', name='on_yComboBox_currentIndexChanged')
    #@qtSlot('int', name='on_colorComboBox_currentIndexChanged')
    @qtSlot()
    def on_updatePushButton_clicked(self):
        ''' Redraw the plot
        '''
        
        self.drawPlot()
        
    def drawPlot(self, index=None):
        ''' Draw a scatter plot
        '''
        
        self.axes.clear()
        x = self.ui.xComboBox.currentIndex()
        y = self.ui.yComboBox.currentIndex()
        c = self.ui.colorComboBox.currentIndex()
        s = self.ui.subsetComboBox.currentIndex()
        if s > 0:
            s = self.headermap[str(self.ui.subsetComboBox.currentText())]
            if index is not None and hasattr(index, 'text'):
                self.clear()
                sval = float(index.text())
                if index.checkState() == QtCore.Qt.Checked:
                    self.selected = numpy.logical_or(self.selected, self.data[:, s]==sval)
                else:
                    self.selected = numpy.logical_and(self.selected, self.data[:, s]!=sval)
            else:
                for i in xrange(self.subsetListModel.rowCount()):
                    sval = float(self.subsetListModel.item(i).text())
                    if i == 1:
                        _logger.debug("%d: %f -> %d -- sum: %d -- %s"%(i, sval, self.subsetListModel.item(i).checkState() == QtCore.Qt.Checked, numpy.sum(self.data[:, s-1]==sval), str(self.data[:5, s])))
                    if self.subsetListModel.item(i).checkState() == QtCore.Qt.Checked:
                        self.selected = numpy.logical_or(self.selected, self.data[:, s]==sval)
                    else:
                        self.selected[self.data[:, s]==sval]=0# = numpy.logical_and(self.selected, self.data[:, s-1]!=sval)
                    
        data = self.data[self.selected]
        if len(data) > 0:
            color = 'r'
            cmap = None
            if c > 0:
                color = data[:, c-1]
                if color.max() != color.min():
                    color -= color.min()
                    color /= color.max()
                    #cmap = cm.gist_rainbow #cm.cool
                    cmap = cm.jet #cm.cool #cm.winter
                    groups = numpy.unique(color)
                    if len(groups) < len(self.markers):
                        self.groups=[]
                        for i, g in enumerate(groups):
                            sel = g==color
                            self.groups.append(numpy.argwhere(sel))
                            self.axes.scatter(data[sel, x], data[sel, y], c=cmap(g), picker=5, s=10, marker=self.markers[i], edgecolor = 'face')
                    else:
                        self.groups=None
                        self.axes.scatter(data[:, x], data[:, y], marker='o', cmap=cmap, c=color, picker=5, s=10, edgecolor = 'face')
                    
                else: 
                    self.groups=None
                    color = 'r'
                    self.axes.scatter(data[:, x], data[:, y], marker='+', cmap=cmap, c=color, picker=5, s=10, edgecolor = 'face')
        self.ui.canvas.draw()
        self.drawImages()
    
    def displayImage(self, event_obj):
        ''' Event invoked when user selects a data point
        
        :Parameters:
        
        event_obj : Event
                Mouse click event
        '''
        
        if len(event_obj.ind) > 0:
            xc = self.ui.xComboBox.currentIndex()
            yc = self.ui.yComboBox.currentIndex()
            
            #thisline = event_obj.artist
            #xdata = thisline.get_xdata()
            #ydata = thisline.get_ydata()
            
            x = event_obj.mouseevent.xdata
            y = event_obj.mouseevent.ydata
            #dx = self.data[event.ind, xc] if self.groups is None else self.data[event_obj.ind, xc]
            if self.groups is not None and len(self.groups) > 0:
                
                min_val = (1e20, None)
                for group in self.groups:
                    ind = numpy.asarray([i for i in event_obj.ind if i < len(group)], dtype=numpy.int)
                    if len(ind) == 0: continue
                    ds = numpy.hypot(x-self.data[group[ind], xc], y-self.data[group[ind], yc])
                    if ds.min() < min_val[0]: min_val = (ds.min(), group[ind[ds.argmin()]])
                self.selectedImage = min_val[1]
            else:
                ds = numpy.hypot(x-self.data[event_obj.ind, xc], y-self.data[event_obj.ind, yc]) # if data.shape[1] == 2 else numpy.hypot(x-data[event_obj.ind, 0], y-data[event_obj.ind, 1], z-data[event_obj.ind, 2])
                #ds = numpy.hypot(x-xdata[event_obj.ind], y-ydata[event_obj.ind])
                self.selectedImage = event_obj.ind[ds.argmin()]
            self.drawImages()
    
    @qtSlot(int)
    def on_keepSelectedCheckBox_stateChanged(self, state):
        '''
        '''
        
        self.ui.selectGroupComboBox.setEnabled(state == QtCore.Qt.Checked)
        self.clear()
        self.drawImages()
    
    @qtSlot(int)
    def on_imageCountSpinBox_valueChanged(self, index):
        '''
        '''
        
        self.drawImages(index)
    
    @qtSlot(int)
    def on_imageSepSpinBox_valueChanged(self, index):
        '''
        '''
        
        self.drawImages(index)
    
    @qtSlot(float)
    def on_imageZoomDoubleSpinBox_valueChanged(self, index):
        '''
        '''
        
        self.drawImages(index)
    
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
            _logger.debug("Image selected and kept: %d == %d"%(numpy.sum(self.selected), len(self.data)))
            if self.selectedImage is not None:
                idx = self.selectedImage
                try:
                    if len(idx)==1: idx=idx[0]
                except:pass
                data = self.data[self.selected]
                if self.rtsq is not None:
                    rdata = data[:, self.rtsq]
                else: rdata=None
                data = data[:, (x,y)]
                label = self.label[self.selected]
                neighbors = self.ui.averageCountSpinBox.value()
                if neighbors > 2:
                    off = idx
                    idx = numpy.argsort(numpy.hypot(data[idx, 0]-data[:, 0], data[idx, 1]-data[:, 1]))[:neighbors+1].squeeze()
                    avg = None
                    for i, img in enumerate(iter_images(self.stack_file, label[idx])):
                        if rdata is not None:
                            print label[i], rdata[idx[i], 0], rdata[idx[i], 1], rdata[idx[i], 2]
                            img = rotate.rotate_image(img, rdata[idx[i], 0], rdata[idx[i], 1], rdata[idx[i], 2])
                        if avg is None: avg = img.copy()
                        else: avg += img
                    im = OffsetImage(avg, zoom=zoom, cmap=cm.Greys_r) if img.ndim == 2 else OffsetImage(img, zoom=zoom)
                    ab = AnnotationBbox(im, data[off], xycoords='data', xybox=(radius, 0.), boxcoords="offset points", frameon=False)
                    self.axes.add_artist(ab)
                else:
                    tmp = label[(idx, )].reshape((1, label.shape[1]))
                    for i, img in enumerate(iter_images(self.stack_file, tmp)):
                        im = OffsetImage(img, zoom=zoom, cmap=cm.Greys_r) if img.ndim == 2 else OffsetImage(img, zoom=zoom)
                        ab = AnnotationBbox(im, data[idx], xycoords='data', xybox=(radius, 0.), boxcoords="offset points", frameon=False)
                        self.axes.add_artist(ab)
                    print "idx: ", idx
                    if 1 == 0:
                        if not hasattr(self, 'plot_count'): self.plot_count = 0
                        import pylab, scipy, scipy.stats
                        pylab.figure(self.plot_count)
                        #off = img.shape[0]/2
                        #img = img.ravel()[:len(img.ravel())-len(img.ravel())%off]
                        #img = img.reshape((len(img.ravel())/off, off))
                        
                        rimg=ndimage_utility.rolling_window(img, (5,5), (2,2))
                        rimg = rimg.reshape((rimg.shape[0]*rimg.shape[1], rimg.shape[2]*rimg.shape[3]))
                        print rimg.shape
                        #ccp=numpy.corrcoef(rimg)
                        #pylab.imshow(ccp, cm.jet)
                        if 1 == 1:
                            #self.plot_count += 1
                            #pylab.figure(self.plot_count)
                            d, V = scipy.linalg.svd(rimg, False)[1:]
                            val = d[:2]*numpy.dot(V[:2], rimg.T).T
                            pylab.hist(val[:, 0], int(numpy.sqrt(len(val))))
                            #print scipy.stats.shapiro(val[:, 0])
                            print scipy.stats.normaltest(img.ravel())
                        if 1 == 0:
                            ccp = numpy.corrcoef(img)-numpy.corrcoef(img.T)
                            pylab.imshow(ccp, cm.jet)
                            
                            if 1 == 0:
                                self.plot_count += 1
                                pylab.figure(self.plot_count)
                                d, V = scipy.linalg.svd(img, False)[1:]
                                val = d[:2]*numpy.dot(V[:2], img.T).T
                                pylab.hist(val[:, 0], int(numpy.sqrt(len(val))))
                            if 1 == 0:
                                self.plot_count += 1
                                pylab.figure(self.plot_count)
                                out = numpy.zeros(len(label))
                                for i, img in enumerate(iter_images(self.stack_file, label)):
                                    d, V = scipy.linalg.svd(img, False)[1:]
                                    #val = d[:2]*numpy.dot(V[:2], img.T).T
                                    val = d[:2]*numpy.dot(V[:2], img).T
                                    out[i] = numpy.max(val[:, 0])
                                pylab.hist(out, int(numpy.sqrt(len(out))))
                        
                        pylab.show()
                        self.plot_count += 1
                #self.axes.plot(data[idx, 0], data[idx, 1], 'o', ms=12, alpha=0.4, color='yellow', visible=True)
                self.axes.scatter(data[idx, 0], data[idx, 1], marker='o', s=45, facecolors='none', edgecolors='r', visible=True)
                if self.highlight_index > 0:
                    scol = self.headermap[str(self.ui.selectGroupComboBox.currentText())]
                    print 'scol: ', scol, self.data[0, :5], self.highlight_index-1, self.header, self.ui.selectGroupComboBox.currentText()
                    #scol = self.highlight_index-1
                    hindex = numpy.argwhere(self.data[:, scol] == self.data[idx, scol]).squeeze()
                    print "hindex:", hindex.shape, self.data[idx, scol]
                    for h in hindex:
                        self.axes.scatter(self.data[h, x], self.data[h, y], marker='o', s=45, facecolors='none', edgecolors='r', visible=True)
                            
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
                tmp = label[(idx, )].reshape((1, label.shape[1]))
                for i, img in enumerate(iter_images(self.stack_file, tmp)):
                    im = OffsetImage(img, zoom=zoom, cmap=cm.Greys_r) if img.ndim == 2 else OffsetImage(img, zoom=zoom)
                    ab = AnnotationBbox(im, data[idx], xycoords='data', xybox=(radius, 0.), boxcoords="offset points", frameon=False)
                    self.axes.add_artist(ab)
                #self.axes.plot(data[idx, 0], data[idx, 1], 'o', ms=12, alpha=0.4, color='yellow', visible=True)
                self.axes.scatter(data[idx, 0], data[idx, 1], marker='o', s=45, facecolors='none', edgecolors='r', visible=True)
            else:
                index = plot_random_sample_of_images(self.axes, self.stack_file, self.label, self.data[:, (x,y)], self.selected, radius, n, zoom, idx)
                #self.axes.plot(self.data[index, x], self.data[index, y], 'o', ms=12, alpha=0.4, color='yellow', visible=True)
                self.axes.scatter(self.data[index, x], self.data[index, y], marker='o', s=45, facecolors='none', edgecolors='r', visible=True)
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
    
    if 1 == 1:
        for img in itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, index)):
            yield img
    else:
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
    
    #del ax.lines[:]
    i=0
    while i < len(ax.collections):
        obj = ax.collections[i]
        if hasattr(obj, 'get_sizes') and len(obj.get_sizes()) > 0 and obj.get_sizes()[0]==45:
            ax.collections.remove(obj)
        else: i+=1
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

            
            