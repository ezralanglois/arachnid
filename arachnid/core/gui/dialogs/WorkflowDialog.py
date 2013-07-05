''' Displays an editor to create a workflow from a set of operations.

.. todo ::
    
    Add checkboxes next to workflow items

.. Created on Dec 3, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..util.qt4_loader import QtCore, QtGui, qtSignal
import logging
from pyui.WorkflowDialog import Ui_WorkflowDialog

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class Dialog(QtGui.QDialog):
    ''' Display a workflow chooser
    
    :Parameters:
    
    operations : list
                 List of available operations in the workflow
    parent : QObject
             Parent object of the dialog
    '''
    
    operationsUpdated = qtSignal('PyQt_PyObject')
    
    def __init__(self, operations, parent=None):
        "Initialize the dialog"
        
        QtGui.QDialog.__init__(self, parent)
        self.ui = Ui_WorkflowDialog()
        self.ui.setupUi(self)
        
        for op in operations:
            self.ui.operatorList.addItem(QtGui.QListWidgetItem(op))
            
        self.connect(self.ui.operatorList, QtCore.SIGNAL("itemDoubleClicked(QListWidgetItem *)"), self.addItems)
        self.connect(self.ui.addButton, QtCore.SIGNAL("clicked()"), self.addItems)
        
        self.connect(self.ui.workflowList, QtCore.SIGNAL("itemDoubleClicked(QListWidgetItem *)"), self.removeItems)
        self.connect(self.ui.removeButton, QtCore.SIGNAL("clicked()"), self.removeItems)
        
        self.connect(self.ui.upButton, QtCore.SIGNAL("clicked()"), self.moveOperationUp)
        self.connect(self.ui.downButton, QtCore.SIGNAL("clicked()"), self.moveOperationDown)
        self.connect(self.ui.exitButton, QtCore.SIGNAL("clicked()"), self.done)
    
    def open(self, items=[]):
        '''Open the dialog with the given workflow
        
        :Parameters:
         
         workflow : list
                    List of operations in the workflow1
        '''
        
        self.ui.workflowList.clear()
        for item in items:
            if item != "":
                self.ui.workflowList.addItem(QtGui.QListWidgetItem(item))
        QtGui.QDialog.open(self)
    
    def done(self, mode=1):
        '''Dialog is done
        
        :Parameters:
        
        mode : int
               Dialog accept mode
        '''
        
        if mode > 0:
            operations = []
            for i in xrange(self.ui.workflowList.count()):
                operations.append(str(self.ui.workflowList.item(i).text()))
            #for i in xrange(self.ui.headerTableWidget.rowCount()):
                #if self.ui.headerTableWidget.item(i, 1).checkState() == QtCore.Qt.Unchecked: continue
                #header.append(self.ui.workflowList.item(i, 0).text())
            self.operationsUpdated.emit(operations)
        super(Dialog, self).done(mode)
    
    def addItems(self, items=None):
        '''Add items from the operation list to the workflow list
        
        :Parameters:
        
        items : list
                List of items to add to work flow
        '''
        
        if items is None: 
            items = self.ui.operatorList.selectedItems()
        if not isinstance(items, list): items = [items]
        for item in items:
            self.ui.workflowList.addItem(QtGui.QListWidgetItem(item.text()))
    
    def removeItems(self, items=None):
        '''Remove items from the workflow list
        
        :Parameters:
        
        items : list
                List of items to remove from work flow
        '''
        
        if items is None: 
            items = self.ui.workflowList.selectedItems()
        if not isinstance(items, list): items = [items]
        for item in items:
            self.ui.workflowList.takeItem(self.ui.workflowList.row(item))
    
    def moveOperationUp(self, listWidget=None):
        '''Move an operation in the specified list up (default is workflow list).
        
        :Parameters:
        
        listWidget : QListWidget
                     List widget
        '''
        
        if listWidget is None:
            listWidget = self.ui.workflowList
        if listWidget.count() < 2: return
        items = listWidget.selectedItems()
        if not isinstance(items, list): items = [items]
        rows = [listWidget.row(item) for item in items]
        tot = listWidget.count()-1
        for r in rows:
            t = r-1 if r > 0 else tot
            _logger.debug("Row %d --> %d -- of %d"%(r, t, tot))
            temp = listWidget.item(t).text()
            listWidget.item(t).setText(listWidget.item(r).text())
            listWidget.item(r).setText(temp)
        if len(rows) == 1:
            listWidget.setItemSelected(listWidget.item(t), True)
    
    def moveOperationDown(self, listWidget=None):
        '''Move an operation in the specified list down (default is workflow list).
        
        :Parameters:
        
        listWidget : QListWidget
                     List widget
        '''
        
        if listWidget is None:
            listWidget = self.ui.workflowList
        if listWidget.count() < 2: return
        items = listWidget.selectedItems()
        if not isinstance(items, list): items = [items]
        rows = [listWidget.row(item) for item in items]
        tot = listWidget.count()-1
        for r in rows:
            t = r+1 if r < tot else 0
            _logger.debug("Row %d --> %d -- of %d"%(r, t, tot))
            temp = listWidget.item(t).text()
            listWidget.item(t).setText(listWidget.item(r).text())
            listWidget.item(r).setText(temp)
        if len(rows) == 1:
            listWidget.setItemSelected(listWidget.item(t), True)
        
    
        
