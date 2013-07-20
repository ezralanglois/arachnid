''' Graphical user interface for screening images

.. Created on Jul 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ImageViewer import MainWindow as ImageViewerWindow
from util.qt4_loader import QtGui,QtCore,qtSlot

class MainWindow(ImageViewerWindow):
    ''' Main window display for the plotting tool
    '''
    
    def __init__(self, parent=None):
        "Initialize screener window"
        
        ImageViewerWindow.__init__(self, parent)
        
        
    def setup(self):
        ''' Display specific setup
        '''
        
        self.connect(self.ui.imageListView.selectionModel(), QtCore.SIGNAL("selectionChanged(const QItemSelection &, const QItemSelection &)"), self.onSelectionChanged)
        self.ui.imageListView.setStyleSheet('QListView::item:selected{ color: #008000; border: 3px solid #6FFF00; }')
    
    # Slots
    
    # Custom Slots
    
    def onSelectionChanged(self, selected, deselected):
        ''' Called when the list selection has changed
        
        :Parameters:
        
        selection : QItemSelection
                    List of selection items in the list
        deselected : QItemSelection
                     List of deselected items in the list
        '''
        
        print selected