''' Graphical user interface for plotting points and displaying corresponding images

.. Created on Jul 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''


from util.qt4_loader import QtGui,QtCore,qtSlot
if not hasattr(QtCore, 'pyqtSlot'):
    import matplotlib
    matplotlib.rcParams['backend.qt4']='PySide'

'''
try:
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
except:
    print "Cannot import offset, upgrade matplotlib"
'''
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
#import matplotlib.cm as cm, matplotlib.lines
#

from pyui.PlotViewer import Ui_MainWindow
from util import qimage_utility
from ..metadata import spider_utility #, format
from ..image import ndimage_utility, ndimage_file, ndimage_interpolate
import os, itertools #, glob, numpy
import logging
#import property


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
        self.coordinates_file = ""
        self.stack_file = ""
        self.inifile = "" #'ara_view.ini'
        
        # Setup Plotting View
        self.fig = Figure((6.0, 4.0), dpi=self.ui.dpiSpinBox.value())
        self.ui.canvas = FigureCanvas(self.fig)
        self.ui.canvas.setParent(self.ui.centralwidget)
        self.axes = self.fig.add_subplot(111)
        self.ui.centralHLayout.addWidget(self.ui.canvas)
        self.ui.centralHLayout.setStretchFactor(self.ui.canvas, 4)
        
        # Setup Navigation Tool Bar
        self.ui.mpl_toolbar = NavigationToolbar(self.ui.canvas, self)
        self.ui.mpl_toolbar.hide()
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.ui.mpl_toolbar)
        if not hasattr(self.ui.mpl_toolbar, 'edit_parameters'):
            self.ui.toolBar.removeAction(self.ui.actionShow_Options)
        
        
        # Custom Actions
        
        #action = self.ui.dockWidget.toggleViewAction()
        #icon8 = QtGui.QIcon()
        #icon8.addPixmap(QtGui.QPixmap(":/mini/mini/application_side_list.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        #action.setIcon(icon8)
        #self.ui.toolBar.insertAction(self.ui.actionHelp, action)
        
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
    
    # Slots for GUI
    
    #    Matplotlib actions
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
    
    def addToolTipImage(self, imgname, item):
        '''
        '''
        
        if imgname[1] == 0 and self.advanced_settings.second_image != "":
            second_image = spider_utility.spider_filename(self.advanced_settings.second_image, imgname[0])
            if os.path.exists(second_image):
                img = read_image(second_image)
                if hasattr(img, 'ndim'):
                    nstd = self.advanced_settings.second_nstd
                    bin_factor = self.advanced_settings.second_downsample
                    img = ndimage_utility.replace_outlier(img, nstd, nstd, replace='mean')
                    if bin_factor > 1.0: img = ndimage_interpolate.interpolate(img, bin_factor, self.advanced_settings.downsample_type)
                    qimg = qimage_utility.numpy_to_qimage(img)
                else:
                    qimg = img.convertToFormat(QtGui.QImage.Format_Indexed8)
                item.setToolTip(qimage_utility.qimage_to_html(qimg))
    
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
        
        
def read_image(filename, index=None):
    '''
    '''
    
    qimg = QtGui.QImage()
    if qimg.load(filename): return qimg
    return ndimage_utility.normalize_min_max(ndimage_file.read_image(filename, index))
        

def iter_images(files, index):
    ''' Wrapper for iterate images that support color PNG files
    
    :Parameters:
    
    filename : str
               Input filename
    
    :Returns:
    
    img : array
          Image array
    '''
    
    for img in itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(files, index)):
        yield img
            

    