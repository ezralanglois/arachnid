''' Graphical user interface for plotting points and displaying corresponding images

.. Created on Jul 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

from util.qt4_loader import QtGui,QtCore,qtSlot
if not hasattr(QtCore, 'pyqtSlot'):
    import matplotlib
    matplotlib.rcParams['backend.qt4']='PySide'


try:
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
except:
    hdlr=logging.StreamHandler()
    _logger.addHandler(hdlr)
    _logger.warn("Cannot import offset, upgrade matplotlib")
    _logger.removeHandler(hdlr)

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.cm as cm #, matplotlib.lines
import property
#

from pyui.PlotViewer import Ui_MainWindow
#from util import qimage_utility
from ..metadata import format #, spider_utility #, format
from ..image import ndimage_utility, ndimage_file, analysis, rotate, ndimage_interpolate, ndimage_filter
import os, itertools, numpy #, glob
#import property




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
        self.coordinates_files = []
        self.stack_file = ""
        self.inifile = "" #'ara_view.ini'
        self.data=None
        self.header=None
        self.group_indices=None
        self.label_cols=[]
        self.rtsq_cols=[]
        self.select_col=-1
        self.markers=['s', 'o', '^', '>', 'v', 'd', 'p', 'h', '8', '+', 'x']
        self.selectedImage = None
        
        # Setup Plotting View
        self.fig = Figure((6.0, 4.0))#, dpi=self.ui.dpiSpinBox.value())
        self.ui.canvas = FigureCanvas(self.fig)
        self.ui.canvas.setParent(self.ui.centralwidget)
        self.axes = self.fig.add_subplot(111)
        self.ui.centralHLayout.addWidget(self.ui.canvas)
        self.ui.centralHLayout.setStretchFactor(self.ui.canvas, 4)
        
        #self.ui.canvas.mpl_connect('motion_notify_event', self.onHover)
        self.ui.canvas.mpl_connect('pick_event', self.displayLabel)
        self.annotation=None
        
        # Setup Navigation Tool Bar
        self.ui.mpl_toolbar = NavigationToolbar(self.ui.canvas, self)
        self.ui.mpl_toolbar.hide()
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.ui.mpl_toolbar)
        if not hasattr(self.ui.mpl_toolbar, 'edit_parameters'):
            self.ui.toolBar.removeAction(self.ui.actionShow_Options)
        
        
        # Custom Actions
        self.ui.toggleImageDockAction = self.ui.imageDockWidget.toggleViewAction()
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/mini/mini/image.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ui.toggleImageDockAction.setIcon(icon8)
        self.ui.toolBar.insertAction(self.ui.actionShow_Options, self.ui.toggleImageDockAction)
        self.ui.imageDockWidget.hide()
        self.ui.toggleImageDockAction.setEnabled(False)
        
        action = self.ui.plotDockWidget.toggleViewAction()
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/mini/mini/chart_line.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action.setIcon(icon8)
        self.ui.toolBar.insertAction(self.ui.actionShow_Options, action)
        
        
        self.ui.toggleAdvancedDockAction = self.ui.advancedDockWidget.toggleViewAction()
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/mini/mini/cog_edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ui.toggleAdvancedDockAction.setIcon(icon8)
        self.ui.toolBar.insertAction(self.ui.actionShow_Options, self.ui.toggleAdvancedDockAction)
        self.ui.advancedDockWidget.hide()
        
        
        action = self.ui.fileDockWidget.toggleViewAction()
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/mini/mini/folder_explore.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action.setIcon(icon8)
        self.ui.toolBar.insertAction(self.ui.actionShow_Options, action)
        
        
        # Create advanced settings
        
        property.setView(self.ui.advancedSettingsTreeView)
        self.advanced_settings, self.advanced_names = self.ui.advancedSettingsTreeView.model().addOptionList(self.advancedSettings())
        self.ui.advancedSettingsTreeView.setStyleSheet('QTreeView::item[readOnly="true"]{ color: #000000; }')
        
        for i in xrange(self.ui.advancedSettingsTreeView.model().rowCount()-1, 0, -1):
            if self.ui.advancedSettingsTreeView.model().index(i, 0).internalPointer().isReadOnly(): # Hide widget items (read only)
                self.ui.advancedSettingsTreeView.setRowHidden(i, QtCore.QModelIndex(), True)
        
        # Subset List
        self.subsetListModel = QtGui.QStandardItemModel()
        self.ui.subsetListView.setModel(self.subsetListModel)
        
        # File List
        self.fileListModel = QtGui.QStandardItemModel()
        self.ui.fileTableView.setModel(self.fileListModel)
        self.fileListModel.setHorizontalHeaderLabels(['file', 'items'])
        self.connect(self.ui.fileTableView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.openSelectedFile)
        
        # Plot
        
        # Create advanced settings
        
        #property.setView(self.ui.advancedSettingsTreeView)
        #self.advanced_settings, self.advanced_names = self.ui.advancedSettingsTreeView.model().addOptionList(self.advancedSettings())
        #self.ui.advancedSettingsTreeView.setStyleSheet('QTreeView::item[readOnly="true"]{ color: #000000; }')
        
        #for i in xrange(self.ui.advancedSettingsTreeView.model().rowCount()-1, 0, -1):
        #    if self.ui.advancedSettingsTreeView.model().index(i, 0).internalPointer().isReadOnly(): # Hide widget items (read only)
        #        self.ui.advancedSettingsTreeView.setRowHidden(i, QtCore.QModelIndex(), True)
        
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
    
    def advancedSettings(self):
        ''' Get a list of advanced settings
        '''
        
        return [ 
               dict(downsample_type=('bilinear', 'ft', 'fs'), help="Choose the down sampling algorithm ranked from fastest to most accurate"),
               dict(bin_factor=1.0, help="Factor to downsample image"),
               dict(gaussian_low_pass=0.0, help="Radius for Gaussian low pass filter"),
               dict(gaussian_high_pass=0.0, help="Radius for Gaussian high pass filter"),
               dict(trans_scale=0.0, help="Value to scale translations (usually pixel size)"),
               
               #dict(film=False, help="Set true to disable contrast inversion"),
               #dict(zoom=self.ui.imageZoomDoubleSpinBox.value(), help="Zoom factor where 1.0 is original size", gui=dict(readonly=True)),
               ]
    
    # Slots for GUI
    
    #    Matplotlib actions
        
    @qtSlot()
    def on_actionOpen_triggered(self):
        ''' Called when someone clicks the Open Button
        '''
        
        if self.ui.actionZoom.isChecked(): self.ui.actionZoom.trigger()
        if self.ui.actionPan.isChecked(): self.ui.actionPan.trigger()
        files = QtGui.QFileDialog.getOpenFileNames(self.ui.centralwidget, self.tr("Open a set of images or documents"), self.lastpath)
        if isinstance(files, tuple): files = files[0]
        if len(files) > 0:
            self.lastpath = os.path.dirname(str(files[0]))
            self.openFiles(files)
    
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
    def on_actionRefresh_triggered(self):
        ''' Redraw the plot
        '''
        
        self.drawPlot()
    
    @qtSlot(int)
    def on_subsetComboBox_currentIndexChanged(self, index):
        ''' Called when the user wants to plot only a subset of the data
        
        :Parameters:
        
        index : int
                New index in the subset combobox
        '''
        
        if index > 0:
            index=self.ui.subsetComboBox.itemData(index)
            vals = [str(v) for v in numpy.unique(self.data[:, index])]
        else: vals = []
        
        self.subsetListModel.clear()
        for name in vals:
            item = QtGui.QStandardItem(name)
            item.setCheckState(QtCore.Qt.Checked)
            item.setCheckable(True)
            self.subsetListModel.appendRow(item)
    
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
    def on_imageCountSpinBox_valueChanged(self, index):
        '''
        '''
        
        self.drawImages()
    
    @qtSlot(int)
    def on_imageSepSpinBox_valueChanged(self, index):
        '''
        '''
        
        self.drawImages()
    
    @qtSlot(float)
    def on_imageZoomDoubleSpinBox_valueChanged(self, index):
        '''
        '''
        
        self.drawImages()
        
    @qtSlot(int)
    def on_keepSelectedCheckBox_stateChanged(self, state):
        '''
        '''
        
        self.ui.selectGroupComboBox.setEnabled(state == QtCore.Qt.Checked)
        self.clear()
        self.drawImages()
    
    # Settings controls for persistance
    
    def getSettings(self):
        ''' Get the settings object
        '''
        
        '''
        return QtCore.QSettings(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, "Arachnid", "ImageView")
        '''
        
        if self.inifile == "": return None
        return QtCore.QSettings(self.inifile, QtCore.QSettings.IniFormat)
    
    def saveSettings(self):
        ''' Save the settings of widgets
        '''
        
        settings = self.getSettings()
        if settings is None: return
        settings.setValue('main_window/geometry', self.saveGeometry())
        settings.setValue('main_window/windowState', self.saveState())
        settings.beginGroup('Advanced')
        self.ui.advancedSettingsTreeView.model().saveState(settings)
        settings.endGroup()
        
    def loadSettings(self):
        ''' Load the settings of widgets
        '''
        
        settings = self.getSettings()
        if settings is None: return
        self.restoreGeometry(settings.value('main_window/geometry'))
        self.restoreState(settings.value('main_window/windowState'))
        settings.beginGroup('Advanced')
        self.ui.advancedSettingsTreeView.model().restoreState(settings) #@todo - does not work!
        settings.endGroup()
    
    # Open Files
    
    def openSelectedFile(self, selected, deselected):
        '''
        '''
        
        filename = selected.indexes()[0].data(QtCore.Qt.UserRole+1)
        self.openFile(filename)
    
    def openFile(self, filename):
        ''' Open a coordinate file
        
        :Parameters:
        
        filename : str
                   Input filename
        '''
        
        self.data, self.header = format.read(filename, ndarray=True)
        self.label_cols=[]
        self.rtsq_cols=[]
        self.select_col=-1
        try: self.label_cols.append(self.header.index('fileid'))
        except: pass
        try: self.label_cols.append(self.header.index('id'))
        except: pass
        try: self.rtsq_cols.extend([self.header.index('psi'),self.header.index('tx'),self.header.index('ty'),self.header.index('mirror')])
        except: 
            try: self.rtsq_cols.extend([self.header.index('psi'),self.header.index('tx'),self.header.index('ty')])
            except: pass
        try: self.select_col = self.header.index('select')
        except: pass
        
        skip = set(self.label_cols+self.rtsq_cols+[self.select_col])
        updateComboBox(self.ui.xComboBox, self.header, skip, index='c0')
        updateComboBox(self.ui.yComboBox, self.header, skip, index='c1')
        skip = set(self.label_cols)
        updateComboBox(self.ui.colorComboBox, self.header, skip, 'select', first='None')
        skip = set(self.label_cols+self.rtsq_cols+[self.select_col]+non_int_columns(self.data))
        updateComboBox(self.ui.subsetComboBox, self.header, skip, first='None')
        updateComboBox(self.ui.selectGroupComboBox, self.header, skip, first='None')
        self.ui.selectGroupComboBox.setEnabled(False)
        self.clear()
        self.drawPlot()
    
    def openFiles(self, files):
        ''' Open a collection of files, sort by content type
        
        :Parameters:
        
        files : list
                List of input files
        '''
        
        coordfile=None
        for filename in files:
            filename = str(filename)
            if not ndimage_file.spider.is_readable(filename) and not ndimage_file.mrc.is_readable(filename) and  format.is_readable(filename):
                nameItem = QtGui.QStandardItem(os.path.basename(filename))
                nameItem.setData(filename)
                nameItem.setToolTip(filename)
                countItem = QtGui.QStandardItem(str(len(format.read(filename))))
                countItem.setData(filename)
                countItem.setToolTip(filename)
                self.fileListModel.appendRow([nameItem, countItem])
                self.coordinates_files.append(filename)
                if coordfile is None: coordfile=nameItem #coordfile=filename
            else:
                self.stack_file = filename
        self.coordinates_files = list(set(self.coordinates_files))
        if coordfile is not None:
            self.ui.fileTableView.selectionModel().setCurrentIndex(coordfile.index(), QtGui.QItemSelectionModel.Select|QtGui.QItemSelectionModel.Rows)
            #self.openFile(coordfile)
        if self.stack_file != "":
            self.ui.toggleImageDockAction.setEnabled(True)
            self.ui.imageDockWidget.show()
            self.ui.canvas.mpl_connect('pick_event', self.displayImage)
            
    def dataSubset(self):
        '''
        '''
        
        s = self.ui.subsetComboBox.itemData(self.ui.subsetComboBox.currentIndex())
        if s > -1:
            selected = numpy.ones(self.data.shape[0], dtype=numpy.bool)
            for i in xrange(self.subsetListModel.rowCount()):
                sval = float(self.subsetListModel.item(i).text())
                if self.subsetListModel.item(i).checkState() == QtCore.Qt.Checked:
                    selected = numpy.logical_or(selected, self.data[:, s]==sval)
                else:
                    selected[self.data[:, s]==sval]=0
            data = self.data[selected]
        else:
            data = self.data
        return data
    
    # Plot Points
    def drawPlot(self, index=None):
        ''' Draw a scatter plot
        '''
        
        self.axes.clear()
        data = self.dataSubset()
        self.plotPointsAsScatter(data)
        idx = self.drawImages(data, False)
        self.highlightPoints(data, idx, False)
        self.annotation=None
        self.ui.canvas.draw()
    
    def plotPointsAsScatter(self, data=None, use_markers=False, markersize=10, pickersize=5):
        '''
        '''
        
        if data is None: data = self.dataSubset()
        x = self.ui.xComboBox.itemData(self.ui.xComboBox.currentIndex())
        y = self.ui.yComboBox.itemData(self.ui.yComboBox.currentIndex())
        c = self.ui.colorComboBox.itemData(self.ui.colorComboBox.currentIndex())
        color='r'
        marker=self.markers[1]
        cmap = None
        self.group_indices=None
        if c > -1: 
            color = data[:, c].copy()
            if color.max() != color.min():
                color -= color.min()
                color /= color.max()
                cmap = cm.jet
            else: color='r'
        
        if cmap is not None and use_markers:
            groups = numpy.unique(color)
            self.group_indices=[]
            for i, (g, marker) in enumerate(zip(groups, itertools.cycle(self.markers))):
                sel = g==color
                self.group_indices.append(numpy.argwhere(sel))
                self.axes.scatter(data[sel, x], data[sel, y], c=cmap(g), picker=pickersize, s=markersize, marker=marker, edgecolor = 'face')
        else:
            self.axes.scatter(data[:, x], data[:, y], c=color, cmap=cmap, picker=pickersize, s=markersize, marker=marker, edgecolor = 'face')
    
    def highlightPoints(self, data, idx, draw=True):
        '''
        '''
        
        scol = self.ui.selectGroupComboBox.currentIndex()
        if scol == 0 or idx is None: return
        scol = self.ui.selectGroupComboBox.itemData(scol)
        
        x = self.ui.xComboBox.itemData(self.ui.xComboBox.currentIndex())
        y = self.ui.yComboBox.itemData(self.ui.yComboBox.currentIndex())
        if hasattr(idx, '__iter__'): idx=idx[0]
        hindex = numpy.argwhere(data[:, scol] == data[idx, scol]).squeeze()
        for h in hindex:
            self.axes.scatter(data[h, x], data[h, y], marker='o', s=45, facecolors='none', edgecolors='r', visible=True)
        if draw: self.ui.canvas.draw()
    
    # Plot Images
    def clear(self):
        ''' Clear the plot of images
        '''
        
        clearImages(self.axes)
        self.selectedImage = None
        self.ui.canvas.draw()
        
    def drawImages(self, data=None, draw=True):
        ''' Draw images on the plot
        '''
        
        idx=None
        if data is None: data = self.dataSubset()
        if self.ui.keepSelectedCheckBox.checkState() == QtCore.Qt.Checked:
            if self.selectedImage is not None:
                neighbors = self.ui.averageCountSpinBox.value()
                if neighbors > 2: idx=self.drawAverageImages(data, neighbors)
                else: idx=self.drawSelectedImages(data)
                x = self.ui.xComboBox.itemData(self.ui.xComboBox.currentIndex())
                y = self.ui.yComboBox.itemData(self.ui.yComboBox.currentIndex())
                self.axes.scatter(data[idx, x], data[idx, y], marker='o', s=45, facecolors='none', edgecolors='r', visible=True)
        else:
            self.drawRandomImages(data)
        if draw: self.ui.canvas.draw()
        return idx
    
    def drawSelectedImages(self, data):
        ''' Draw a random subset of images on the plot
        '''
        
        x = self.ui.xComboBox.itemData(self.ui.xComboBox.currentIndex())
        y = self.ui.yComboBox.itemData(self.ui.yComboBox.currentIndex())
        zoom = self.ui.imageZoomDoubleSpinBox.value()
        radius = self.ui.imageSepSpinBox.value()
        off = self.selectedImage
        align = data[(off, ), self.rtsq_cols].reshape((1, len(self.rtsq_cols))) if len(self.rtsq_cols) > 2 else None
        tmp = data[(off, ), self.label_cols].reshape((1, len(self.label_cols)))
        for i, img in enumerate(iter_images(self.stack_file, tmp, align, **vars(self.advanced_settings))):
            im = OffsetImage(img, zoom=zoom, cmap=cm.Greys_r) if img.ndim == 2 else OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(im, data[off, (x,y)], xycoords='data', xybox=(radius, 0.), boxcoords="offset points", frameon=False)
            self.axes.add_artist(ab)
        return off
    
    def drawAverageImages(self, data, neighbors):
        ''' Draw a random subset of images on the plot
        '''
        
        x = self.ui.xComboBox.itemData(self.ui.xComboBox.currentIndex())
        y = self.ui.yComboBox.itemData(self.ui.yComboBox.currentIndex())
        zoom = self.ui.imageZoomDoubleSpinBox.value()
        radius = self.ui.imageSepSpinBox.value()
        off = self.selectedImage
        idx = numpy.argsort(numpy.hypot(data[off, x]-data[:, x], data[off, y]-data[:, y]))[:neighbors+1].squeeze()
        avg = None
        data2 = data[idx]
        sidx = numpy.argsort(data2[:, self.label_cols[0]])
        data2 = data2[sidx]
        align = data2[:, self.rtsq_cols] if len(self.rtsq_cols) > 2 else None
        print 'averaging'
        for img in iter_images(self.stack_file, data2[:, self.label_cols], align, **vars(self.advanced_settings)):
            if avg is None: avg = img.copy()
            else: avg += img
        avg = ndimage_utility.normalize_min_max(avg)
        im = OffsetImage(avg, zoom=zoom, cmap=cm.Greys_r) if img.ndim == 2 else OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, data[off, (x,y)], xycoords='data', xybox=(radius, 0.), boxcoords="offset points", frameon=False)
        self.axes.add_artist(ab)
        return idx
    
    def drawRandomImages(self, data):
        ''' Draw a random subset of images on the plot
        '''
        
        clearImages(self.axes)
        n = self.ui.imageCountSpinBox.value()
        total = len(data)
        if n > total: n = total
        if n < 2: return
        idx = self.selectedImage if self.selectedImage is not None else numpy.random.randint(0, data.shape[0])
        x = self.ui.xComboBox.itemData(self.ui.xComboBox.currentIndex())
        y = self.ui.yComboBox.itemData(self.ui.yComboBox.currentIndex())
        zoom = self.ui.imageZoomDoubleSpinBox.value()
        radius = self.ui.imageSepSpinBox.value()
        align = data[:, self.rtsq_cols] if len(self.rtsq_cols) > 2 else None
        index = plot_random_sample_of_images(self.axes, self.stack_file, data[:, self.label_cols], data[:, (x,y)], align, radius, n, zoom, idx, **vars(self.advanced_settings))
        self.axes.scatter(data[index, x], data[index, y], marker='o', s=45, facecolors='none', edgecolors='r', visible=True)
    
    def displayImage(self, event_obj):
        ''' Event invoked when user selects a data point
        
        :Parameters:
        
        event_obj : Event
                Mouse click event
        '''
        
        if not hasattr(event_obj, 'ind'): return
        if len(event_obj.ind) == 0: return
        xc = self.ui.xComboBox.itemData(self.ui.xComboBox.currentIndex())
        yc = self.ui.yComboBox.itemData(self.ui.yComboBox.currentIndex())
        x = event_obj.mouseevent.xdata
        y = event_obj.mouseevent.ydata
        data = self.dataSubset()
        if self.group_indices is not None and len(self.group_indices) > 0:
            min_val = (1e20, None)
            for group in self.group_indices:
                ind = numpy.asarray([i for i in event_obj.ind if i < len(group)], dtype=numpy.int)
                if len(ind) == 0: continue
                ds = numpy.hypot(x-data[group[ind], xc], y-data[group[ind], yc])
                if ds.min() < min_val[0]: min_val = (ds.min(), group[ind[ds.argmin()]])
            self.selectedImage = min_val[1]
        else:
            ds = numpy.hypot(x-data[event_obj.ind, xc], y-data[event_obj.ind, yc])
            self.selectedImage = event_obj.ind[ds.argmin()]
        self.displayLabel(event_obj, False)
        self.drawImages()
    
    # Plot label
    def displayLabel(self, event_obj, repaint=True):
        '''
        '''
        
        if hasattr(event_obj, 'ind') and len(event_obj.ind) > 0:
            data = self.dataSubset()
            xc = self.ui.xComboBox.itemData(self.ui.xComboBox.currentIndex())
            yc = self.ui.yComboBox.itemData(self.ui.yComboBox.currentIndex())
            x = event_obj.mouseevent.xdata
            y = event_obj.mouseevent.ydata
            if self.group_indices is not None and len(self.group_indices) > 0:
                min_val = (1e20, None)
                for group in self.group_indices:
                    ind = numpy.asarray([i for i in event_obj.ind if i < len(group)], dtype=numpy.int)
                    if len(ind) == 0: continue
                    ds = numpy.hypot(x-data[group[ind], xc], y-data[group[ind], yc])
                    if ds.min() < min_val[0]: min_val = (ds.min(), group[ind[ds.argmin()]])
                idx = min_val[1]
            else: 
                ds = numpy.hypot(x-data[event_obj.ind, xc], y-data[event_obj.ind, yc])
                idx = event_obj.ind[ds.argmin()]
            
            text = " ".join([str(v) for v in data[idx, self.label_cols]])
            if  self.annotation is None:
                 self.annotation = self.axes.annotate(text, xy=(x,y),  xycoords='data',
                            xytext=(-15, 15), textcoords='offset points',
                            arrowprops=dict(arrowstyle="->")
                            )
                 self.annotation.draggable()
            else:
                self.annotation.xy = x,y
                self.annotation.set_text(text)
                self.annotation.set_visible(True)
            if repaint: self.ui.canvas.draw()
        
def plot_random_sample_of_images(ax, stack_file, label, xy, align, radius, n, zoom, keep=None, **extra):
    '''
    '''
    
    index = numpy.arange(len(xy), dtype=numpy.int)
    numpy.random.shuffle(index)
    off = numpy.argwhere(index==keep)
    if off.shape[0] > 0:
        off = off.squeeze()
        index[off]=index[0]
        index[0]=keep
    assert(n>0)
    
    vals = nonoverlapping_subset(ax, xy[index], radius, n)
    index = index[vals]
    label = label[index]
    sidx = numpy.argsort(label[:, 0])
    label = label[sidx]
    if align is not None: 
        align = align[index]
        align = align[sidx]
    
    
    for i, img in enumerate(iter_images(stack_file, label, align, **extra)):
        im = OffsetImage(img, zoom=zoom, cmap=cm.Greys_r) if img.ndim == 2 else OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, xy[index[sidx[i]]], xycoords='data', xybox=(radius, 0.), boxcoords="offset points", frameon=False)
        ax.add_artist(ab)
    return index

def non_int_columns(data):
    '''
    '''
    
    subset=[]
    for i in xrange(data.shape[1]):
        if not numpy.alltrue(data[:, i]==data[:, i].astype(numpy.int)):
            subset.append(i)
    return subset

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

def updateComboBox(combo, values, skip=set(), index=None, first=None):
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
    if index is not None and not isinstance(index, int):
        if index not in set(values): index=None
    if index is None and combo.count() > 0:
        index = combo.currentIndex()
    combo.clear()
    j=0
    if first is not None: 
        combo.addItem(first, -1)
        j+=1
    
    for i, v in enumerate(values):
        if i in skip: continue
        combo.addItem(v, i)
        if index == v: index = j
        j+=1
    combo.blockSignals(False)
    if index is not None and isinstance(index, int): 
        combo.setCurrentIndex(index)

def read_image(filename, index=None):
    '''
    '''
    
    qimg = QtGui.QImage()
    if qimg.load(filename): return qimg
    return ndimage_utility.normalize_min_max(ndimage_file.read_image(filename, index))
        

def iter_images(files, index, align=None, bin_factor=1.0, downsample_type='bilinear', gaussian_high_pass=0.0, gaussian_low_pass=0.0, trans_scale=0.0, **extra):
    ''' Wrapper for iterate images that support color PNG files
    
    :Parameters:
    
    filename : str
               Input filename
    
    :Returns:
    
    img : array
          Image array
    '''
    
    if hasattr(index, 'ndim'): index = numpy.asarray(index, dtype=numpy.int)
    i=0
    #for img in itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(files, index)):
    for img in ndimage_file.iter_images(files, index):
        if align is not None:
            if trans_scale > 0:
                print trans_scale, align[i, 0], align[i, 1]/trans_scale, align[i, 2]/trans_scale, align[i,3]
                img = rotate.rotate_image(img, align[i, 0], align[i, 1]/trans_scale, align[i, 2]/trans_scale)
            else:
                img = rotate.rotate_image(img, align[i, 0], align[i, 1], align[i, 2])
            if len(align[i]) > 3 and align[i,3] > 180: img = ndimage_utility.mirror(img)
            
            i+=1
        if bin_factor > 1.0: img = ndimage_interpolate.interpolate(img, bin_factor, downsample_type)
        if gaussian_high_pass > 0.0:
            img=ndimage_filter.filter_gaussian_highpass(img, gaussian_high_pass)
        if gaussian_low_pass > 0.0:
            img=ndimage_filter.filter_gaussian_lowpass(img, gaussian_low_pass)
        img = ndimage_utility.normalize_min_max(img)
        yield img

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




    