''' Defines a set of delegates to display widgets in a table/tree cell

.. Created on Dec 11, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..dialogs.WorkflowDialog import Dialog as WorkflowDialog
from ..util.qt4_loader import QtGui,QtCore, qtSignal
import os, logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class CheckboxWidget(QtGui.QWidget):
    ''' Display a checkbox in a table/tree cell
        
    :Parameters:
    
    parent : QObject
             Parent of the checkbox widget
    '''
    
    def __init__(self, parent=None):
        "Initialize a font dialog"
        
        QtGui.QWidget.__init__(self, parent)
        
        #self.setStyleSheet("QWidget { background-color: White }")
        self.button = QtGui.QCheckBox(self)
        self.spacer = QtGui.QSpacerItem(10, 14, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.formLayout = QtGui.QFormLayout(self)
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.ExpandingFieldsGrow)#FieldsStayAtSizeHint)
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)#QtCore.Qt.AlignLeading|
        self.formLayout.setFormAlignment(QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)#QtCore.Qt.AlignLeading|
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setMargin(0)
        self.formLayout.setObjectName("formLayout")
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.button)
        self.formLayout.setItem(0, QtGui.QFormLayout.FieldRole, self.spacer)
        self.button.setFocusPolicy(QtCore.Qt.StrongFocus)

class DialogWidget(QtGui.QWidget):
    ''' Abtract class to create a button, which displays a dialog
        
    :Parameters:
    
    parent : QObject
             Parent of the checkbox widget
    icon : QIcon
           Icon for the button
    keep_editor : bool
                  Keep the text editor
    '''
    
    editFinished = qtSignal()
    
    def __init__(self, parent=None, icon=None, keep_editor=False):
        "Initialize a font dialog"
        
        QtGui.QWidget.__init__(self, parent)
        
        if icon is None: icon = ":/mini/mini/folder.png"
        self.button = QtGui.QToolButton(self)
        self.action = QtGui.QAction(QtGui.QIcon(icon), "", self)
        self.button.setDefaultAction(self.action)
        
        if keep_editor:
            self.layout = QtGui.QHBoxLayout(self)
            self.layout.setObjectName("dialogLayout")
            self.layout.setContentsMargins(0, 0, 5, 0)
            self.field = QtGui.QLineEdit(self)
            self.layout.addWidget(self.field)
            self.layout.addWidget(self.button)
        else:
            self.spacer = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
            self.formLayout = QtGui.QFormLayout(self)
            self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.ExpandingFieldsGrow)#FieldsStayAtSizeHint)
            self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
            self.formLayout.setFormAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignRight|QtCore.Qt.AlignTop)
            self.formLayout.setContentsMargins(0, 0, 5, 0)
            self.formLayout.setObjectName("formLayout")
            self.formLayout.setItem(0, QtGui.QFormLayout.LabelRole, self.spacer)
            self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.button)
        self.button.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.connect(self.action, QtCore.SIGNAL("triggered()"), self.showDialog)
    
    def showDialog(self):
        ''' Display a dialog (abstract)
        '''
        
        pass
"""
class WorkflowWidget(DialogWidget):
    ''' Create a button and display a workflow dialog on press
        
    :Parameters:
    
    operations : list
                 List of options
    parent : QObject
             Parent of the checkbox widget
    '''
    
    operationsUpdated = qtSignal('PyQt_PyObject')
    
    def __init__(self, operations, parent=None):
        "Initialize a font dialog"
        
        DialogWidget.__init__(self, parent, ':/mini/mini/script_edit.png')
        self.workflow_ops = []
        self.dialog = WorkflowDialog(operations, parent)
        self.connect(self.dialog, QtCore.SIGNAL('operationsUpdated(PyQt_PyObject)'), self.onOperationsUpdated)
    
    def showDialog(self):
        ''' Display a workflow dialog
        '''
        
        self.dialog.open(self.workflow_ops)
    
    def onOperationsUpdated(self, items):
        '''Emit operationsUpdated signal whem the operations dialog 
        is closed.
        
        :Parameters:
        
        items : list
                List of operations
        '''
        
        self.setWorkflow(items)
        self.operationsUpdated.emit(items)
    
    def setWorkflow(self, items):
        ''' Set the list of current operations
        
        :Parameters:
        
        items : list
                List of operations
        '''
        
        self.workflow_ops = items
    
    def workflow(self):
        ''' Get the list of current operations
        
        :Returns:
        
        items : list
                List of operations
        '''
        
        return self.workflow_ops
"""
class FontDialogWidget(DialogWidget):
    ''' Create a button and display font dialog on press
        
    :Parameters:
    
    parent : QObject
             Parent of the checkbox widget
    '''
    
    fontChanged = qtSignal(QtGui.QFont)
    
    def __init__(self, parent=None):
        "Initialize a font dialog"
        
        DialogWidget.__init__(self, parent)
        self.font = QtGui.QFont()
    
    def showDialog(self):
        ''' Display a QFontDialog
        '''
        
        _logger.debug("Show dialog")
        curfont, ok = QtGui.QFontDialog.getFont(self.font, None, "Label Font")
        if ok: 
            self.font = curfont
            self.fontChanged.emit(self.font)
        self.editFinished.emit()
    
    def setCurrentFont(self, font):
        ''' Set the current font
        
        :Parameters:
        
        font : QFont
                Font to display
        '''
        
        self.font = font
    
    def selectedFont(self):
        ''' Get the current font
        
        :Returns:
        
        font : QFont
                Selected font
        '''
        
        return self.font

class FileDialogWidget(DialogWidget):
    ''' Create a button and display file dialog on press
        
    :Parameters:
    
    type : str
           Type of file dialog: open or save
    filter : str
             Semi-colon separated list of file filters
    path : str
           Starting directory for file dialog
    parent : QObject
             Parent of the checkbox widget
    '''
    
    fileChanged = qtSignal(str)
    
    def __init__(self, type, filter="", path="", parent=None):
        "Initialize a font dialog"
        
        DialogWidget.__init__(self, parent, keep_editor=True)
        self.filename = ""
        self.filter = filter
        self.path = path
        self.filetype = type
        self.field.setText(self.filename)
        self.connect(self.field, QtCore.SIGNAL('editingFinished()'), self.updateFilename)
    
    def showDialog(self):
        ''' Display a file dialog
        '''
        
        _logger.debug("Show dialog %s"%self.filetype)
        if self.filetype == 'file-list':
            filenames = QtGui.QFileDialog.getOpenFileNames(None, 'Open files', self.path, self.filter)
            if isinstance(filenames, tuple): filenames = filenames[0]
            filename = ",".join([str(f) for f in filenames])
        elif self.filetype == 'open':
            filename = QtGui.QFileDialog.getOpenFileName(None, 'Open file', self.path, self.filter)
            if isinstance(filename, tuple): filename = filename[0]
        else:
            filename = QtGui.QFileDialog.getSaveFileName(None, 'Save file', self.path, self.filter)
            if isinstance(filename, tuple): filename = filename[0]
        if filename:
            self.filename = filename
            self.fileChanged.emit(self.filename)
        self.editFinished.emit()
    
    def updateFilename(self):
        ''' Update the filename from the line edit
        '''
        
        filename = str(self.field.text())
        if self.filetype == 'file-list' and filename.find(',') != -1:
            filename = filename.split(",")[0]
        if not os.path.isdir(filename):
            self.path = os.path.dirname(str(filename))
        else: self.path = filename
        if self.filetype == 'open' and not os.path.exists(filename) and filename != "":
            self.field.setText("")
            self.showDialog()
        else:
            self.filename = str(self.field.text())
            self.editFinished.emit()
    
    def setCurrentFilename(self, filename):
        ''' Set the current filename
        
        :Parameters:
        
        filename : str
                   Filename to display
        '''
        
        self.filename = str(filename)
        if not os.path.isdir(self.filename):
            self.path = os.path.dirname(str(self.filename))
        else: self.path = self.filename
        self.field.setText(filename)
    
    def selectedFilename(self):
        ''' Get the current filename
        
        :Returns:
        
        filename : str
                   Selected filename
        '''
        
        return self.filename

