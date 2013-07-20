''' Defines a document item model for List, Table or Tree view

.. Created on Jan 5, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..util.qt4_loader import QtCore, qtSignal
#import numpy


class DocumentModel(QtCore.QAbstractTableModel):
    '''This class defines a general model to display a numpy.record in 
    PyQt Lists, Trees and Tables.
    '''
    
    modelUpdated = qtSignal('PyQt_PyObject', 'PyQt_PyObject', int)
    orderUpdated = qtSignal('PyQt_PyObject')
    
    def __init__(self, min_header=[], parent=None):
        "Initialize an abstract meta data model"
        
        QtCore.QAbstractTableModel.__init__(self, parent)
        self.min_header=min_header
        self.clear()
    
    def clear(self):
        '''Clear all project specific data
        '''
        
        self.records = []
        self.header = list(self.min_header)
        self.order = None
        self.currentIds = set()
    
    def data(self, index, role=QtCore.Qt.DisplayRole):
        '''Get data with specified index and type
        
        :Parameters:
        
        index : QModelIndex
                Index in QAbstractTableModel
        role : ItemDataRole
               Role for item data
        
        :Returns:
        
        data : object
               Item data
        '''
        
        if not index.isValid() or not (0 <= index.row() and index.row() < len(self.records)): return None
        if role == QtCore.Qt.DisplayRole:
            if self.order is None:
                return float(self.records[index.row(),index.column()])
            else:
                return float(self.records[self.order[index.row()],index.column()])
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        '''Get header data for specified section, orientation and role
        '''
        
        if (orientation, role) == (QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole):
            if section < len(self.header): return self.header[section]
        return None
    
    def columnCount(self, index=QtCore.QModelIndex()):
        '''Get number of columns
        '''
        
        return len(self.header)
    
    def rowCount(self, index=QtCore.QModelIndex()):
        '''Get number of rows
        '''
        
        return len(self.records)
    
    def sort(self, column, order=QtCore.Qt.AscendingOrder):
        '''Sort nodes based on values in a column
        '''
        
        if column >= len(self.header) or len(self.records) == 0: return
        self.order = self.records[:,column].argsort()
        if order==QtCore.Qt.AscendingOrder: self.order = self.order[::-1]
        QtCore.QAbstractTableModel.reset(self)
        self.orderUpdated.emit(self.order)
    
    def setDocument(self, model, header):
        ''' Set the document model
        '''
        
        self.records = model
        self.header = header
        #if self.order is not None: self.orderUpdated.emit(None)
        self.order = None
        self.modelUpdated.emit(self.records, self.header, self.currentIds)
        QtCore.QAbstractTableModel.reset(self)
    
    def setCurrentId(self, id):
        ''' Set the current Id of the model
        '''
        
        self.currentIds = id
        QtCore.QAbstractTableModel.reset(self)

