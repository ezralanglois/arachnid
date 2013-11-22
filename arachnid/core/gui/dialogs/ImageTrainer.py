''' Main window display for the plotting tool

.. Created on Dec 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from MontageViewer import MainWindow as MontageWindow
from ..util.qt4_loader import QtCore, QtGui, qtSlot
from arachnid.app import autopart
from arachnid.core.metadata import spider_params, format, format_utility
import logging, numpy, os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class MainWindow(MontageWindow):
    ''' Main window display for the plotting tool
    '''
    
    def __init__(self, parent=None):
        "Initialize a basic viewer window"
        
        self.imagepred = []
        self.predfile = 'ara_view_pred.csv'
        MontageWindow.__init__(self, parent)
        
        self.imagesaved = []
        self.alignment = None
        
        # Organize by view and defocus - load pySPIDER alignment file - mirroring
        # command line script for training and prediction
        # Save prediction selections separate
        # Support prediction abstain - draw new random image
        # How to save model?
        
        self.ui.predictAction = QtGui.QAction(QtGui.QIcon(QtGui.QPixmap(":/mini/mini/chart_organisation.png")), "Train", self)
        self.ui.toolBar.insertAction(self.ui.actionHelp, self.ui.predictAction)
        self.ui.predictAction.setEnabled(False)
        self.connect(self.ui.predictAction, QtCore.SIGNAL("triggered()"), self.predict)
        
        self.ui.displayPredictionAction = QtGui.QAction(QtGui.QIcon(QtGui.QPixmap(":/mini/mini/application_view_tile.png")), "Display Prediction", self)
        self.ui.toolBar.insertAction(self.ui.actionHelp, self.ui.displayPredictionAction)
        self.ui.displayPredictionAction.setEnabled(False)
        self.ui.displayPredictionAction.setCheckable(True)
        self.connect(self.ui.displayPredictionAction, QtCore.SIGNAL("triggered(bool)"), self.displayPrediction)
        
        #saved = numpy.loadtxt(self.selectfile, numpy.int, '#', ',')
    
    def loadSelections(self):
        '''
        '''
        
        MontageWindow.loadSelections(self)
        if os.path.exists(self.predfile):
            self.imagepred = numpy.loadtxt(self.predfile, numpy.float, '#', ',')
        
    def displayPrediction(self, checked):
        '''
        '''
        
        if checked:
            self.imagelabel = self.imagepred
        else:
            self.imagelabel = self.imagesaved
    
    def predict(self):
        '''
        '''
        
        train_set = None
        f = QtGui.QFileDialog.getOpenFileName(self.ui.centralwidget, self.tr("Open a SPIDER Params File"), self.lastpath)
        f = str(f[0]) if isinstance(f, tuple) else str(f)
        a = QtGui.QFileDialog.getOpenFileName(self.ui.centralwidget, self.tr("Open a SPIDER Alignment File"), self.lastpath)
        a = str(a[0]) if isinstance(a, tuple) else str(a)
        if f != "":
            last_select = numpy.argwhere(self.imagelabel[:, 2]>0)[-1]
            train_set = numpy.arange(last_select, dtype=numpy.int)
            train_y = self.imagelabel[:, 2]
            param = spider_params.read(f)
            self.imagesaved = self.imagelabel.copy()
            self.imagepred = self.imagelabel.copy()
            param['thread_count'] = 8
            if a != "":
                self.alignment = format.read_alignment(a, ndarray=True)[0]
            self.imagepred[:, 2] = autopart.classify([self.imagefile], self.imagelabel[:, :2], self.alignment, train_set, train_y, resolution=40, **param)
            self.ui.displayPredictionAction.setEnabled(True)
            numpy.savetxt(self.predfile, self.imagepred, delimiter=',', newline='\n')
    
    def onSelectionChanged(self, selected, deselected):
        ''' Called when the list selection has changed
        
        :Parameters:
        
        selection : QItemSelection
                    List of selection items in the list
        deselected : QItemSelection
                     List of deselected items in the list
        '''
        
        MontageWindow.onSelectionChanged(self, selected, deselected)
        if hasattr(self.ui, 'predictAction'):
            if self.selectedCount > 100 and not self.ui.predictAction.isEnabled():
                self.ui.predictAction.setEnabled(True)
            
        
    
    @qtSlot()
    def on_actionHelp_triggered(self):
        ''' Display the help dialog
        '''
        
        box = QtGui.QMessageBox(QtGui.QMessageBox.Information, "Help", '''Tips:
1. Something New
            ''')
        #box.setDetailedText(
        box.exec_()
        
            
