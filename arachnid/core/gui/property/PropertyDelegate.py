''' Delegates the editor for a table/tree cell.

.. note::

    Adopted from http://qt-apps.org/content/show.php/QPropertyEditor?content=68684
    Original Author: Volker Wiendl with Enhancements by Roman alias banal

.. Created on Dec 2, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..util.qt4_loader import QtGui,QtCore
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class PropertyDelegate(QtGui.QItemDelegate):
    ''' Creates the editor widgets for data types
        
    :Parameters:
    
    parent : QObject
             Parent of the Property Delegate
    '''
        
    def __init__(self, parent=None):
        "Initialize the Property Delegate"
        
        QtGui.QItemDelegate.__init__(self, parent)
        self.finishedMapper = QtCore.QSignalMapper(self)
        self.heightHint = 25
    
    def updateEditorGeometry(self, editor, option, index):
        '''Update the geometry of the specified editor
        
        :Parameters:
            
        editor : QWidget
                 Editor widget to display data
        option : QStyleOptionViewItem
                 Used to describe the parameters used to draw an item in a view widget
        index : QModelIndex
                Index of value to edit
        '''
        
        QtGui.QItemDelegate.updateEditorGeometry(self, editor, option, index)
        p = index.internalPointer()
        if p.isBool():
            rect = option.rect
            rect.setLeft(rect.left()+10)
            editor.setGeometry(option.rect)
        
    def sizeHint(self, option, index):
        ''' Get the size hint for the editor widget
        
        :Parameters:
        
        option : QStyleOptionViewItem
                 Used to describe the parameters used to draw an item in a view widget
        index : QModelIndex
                Index of value to edit
        '''
        
        size = QtGui.QItemDelegate.sizeHint(self, option, index)
        size.setHeight(self.heightHint)
        return size
    
    def createEditor(self, parent, option, index):
        ''' Returns the widget used to edit the item specified by index for 
        editing. The parent widget and style option are used to control how 
        the editor widget appears.
        
        :Parameters:
    
        parent : QWidget
                 Parent of created widget
        option : QStyleOptionViewItem
                 Used to describe the parameters used to draw an item in a view widget
        index : QModelIndex
                Index of value to edit
        
        :Returns:
        
        editor : QWidget
                 Widget to edit the cell value
        '''
        
        p = index.internalPointer()
        editor = None
        if not p.isReadOnly():
            editor = p.createEditor(parent, option)
            if editor and editor.metaObject().indexOfSignal("editFinished()") != -1:
                _logger.debug("Created editor - with editFinished %s"%str(editor))
                self.connect(editor, QtCore.SIGNAL("editFinished()"), self.finishedMapper, QtCore.SLOT("map()"))
                self.finishedMapper.setMapping(editor, editor)
            else:
                _logger.debug("Created editor - without editFinished %s"%str(editor))
        if not editor: editor = QtGui.QItemDelegate.createEditor(self, parent, option, index)
        self.parseEditorHints(editor, p)
        return editor
    
    def setEditorData(self, editor, index):
        ''' Sets the data to be displayed and edited by the editor from 
        the data model item specified by the model index
        
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        index : QModelIndex
                Index of value to edit
        '''
        
        self.finishedMapper.blockSignals(True)
        data = index.model().data(index, QtCore.Qt.EditRole)
        success = index.internalPointer().setEditorData(editor, data)
        if not success: 
            QtGui.QItemDelegate.setEditorData(self, editor, index)
        self.finishedMapper.blockSignals(False)
    
    def setModelData(self, editor, model, index):
        ''' Gets data from the editor widget and stores it in the 
        specified model at the item index
        
        :Parameters:
    
        editor : QWidget
                 Editor widget holding new data
        model : QAbstractItemModel
                Model to receive editor data
        index : QModelIndex
                Index of value to edit
        '''
        
        success = False
        data = index.internalPointer().editorData(editor)
        if data is not None: 
            model.setData(index, data, QtCore.Qt.EditRole)
            success = True
        if not success: QtGui.QItemDelegate.setModelData(self, editor, model, index)
    
    def parseEditorHints(self, editor, prop):
        '''Parse editor hints and set as property of the editor
        
        :Parameters:
    
        editor : QWidget
                 Editor widget
        prop : Property
               Current Property
        '''
        
        editorHints = prop.editorHints()
        if editor and editorHints:
            editor.blockSignals(True)
            if isinstance(editorHints, dict):
                for name, value in editorHints.iteritems():
                    if callable(value): value = value(prop.property_obj)
                    editor.setProperty(name, value)
            else:
                rx = QtCore.QRegExp("(.*)(=\\s*)(.*)(;{1})");
                rx.setMinimal(True)
                pos = rx.indexIn(editorHints, 0)
                while pos != -1:
                    editor.setProperty(rx.cap(1).trimmed(), rx.cap(3).trimmed())
                    pos += rx.matchedLength()
                    pos = rx.indexIn(editorHints, pos)
            editor.blockSignals(False)




            




