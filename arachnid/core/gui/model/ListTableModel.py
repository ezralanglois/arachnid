'''
.. Created on Dec 4, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..util.qt4_loader import QtCore

class ListTableModel(QtCore.QAbstractTableModel):#QAbstractItemModel):
    '''
    '''

    def __init__(self, data,header, parent=None):
        '''
        '''
        
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data
        self._header=header
    
    def setData(self, data, header=None):
        '''
        '''
        
        self._data = data
        if header is not None: self._header=header
        self.reset()
        
    def row(self, index):
        '''
        '''
        
        return self._data[index.row()]
    
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        '''Returns the data for the given role and section in the header with the specified orientation
        
        :Parameters:
        
        section : QModelIndex
                  Index of a model item
        orientation : QModelIndex
                      Index of a model item
        role : QModelIndex
               Role of data to return
        
        :Returns:
        
        val : ItemFlags
              Flags for item at index
        '''
        
        if (orientation, role) == (QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole):
            return self._header[section]
        return None
    
    def data(self, index, role = QtCore.Qt.DisplayRole):
        ''' Returns the data stored under the given role for the item referred to by the index
        
        :Parameters:
        
        index : QModelIndex
                Index of a model item
        role : enum
               Data role
        
        :Returns:
        
        val : object
              Data object
        '''
        
        if not index.isValid(): return None
        if role == QtCore.Qt.DisplayRole:
            return str(self._data[index.row()][index.column()])
        return None
    
    def rowCount(self, parent = QtCore.QModelIndex()):
        ''' Get the number of rows for the parent item
        
        :Parameters:
    
        parent : QModelIndex
                 Parent of item
        
        :Returns:
        
        val : int
              Number of rows
        '''
        
        return len(self._data) if self._data is not None else 0
    
    def columnCount(self, parent = QtCore.QModelIndex()):
        ''' Get the number of columns - 2
        
        :Parameters:
    
        parent : QModelIndex
                 Parent of item
        
        :Returns:
        
        val : int
              Return value: 2
        '''
        
        return len(self._data[0]) if self._data is not None and len(self._data) > 0 else 0
    

class CheckListTableModel(ListTableModel):
    '''
    '''
    

    def __init__(self, data,header, parent=None):
        '''
        '''
        
        ListTableModel.__init__(self, data, header, parent)
        self.checked = set()
        if len(data) > 0:
            for i in xrange(len(data[0])): 
                if isinstance(data[0][i], QtCore.Qt.CheckState):
                    self.checked.add(i)
    
    def setData(self, index, value, role=QtCore.Qt.DisplayRole):
        '''
        '''
        
        if not index.isValid(): return False
        
        if role == QtCore.Qt.CheckStateRole and not index.column() in self.checked: print 'here - bad'
        if role == QtCore.Qt.CheckStateRole:
            self._data[index.row()][index.column()] = value
        return False
    
    def data(self, index, role = QtCore.Qt.DisplayRole):
        ''' Returns the data stored under the given role for the item referred to by the index
        
        :Parameters:
        
        index : QModelIndex
                Index of a model item
        role : enum
               Data role
        
        :Returns:
        
        val : object
              Data object
        '''
        
        if not index.isValid(): return None
        if index.column() in self.checked:
            if role == QtCore.Qt.CheckStateRole:
                return self._data[index.row()][index.column()]
            elif role == QtCore.Qt.DisplayRole:
                return None
        return ListTableModel.data(self, index, role)

