'''
This class defines a tree model.

Adopted from http://qt-apps.org/content/show.php/QPropertyEditor?content=68684
Original Author: Volker Wiendl with Enhancements by Roman alias banal

.. Created on Dec 2, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ButtonDelegate import FontDialogWidget, FileDialogWidget, WorkflowWidget #, CheckboxWidget
from PyQt4 import QtGui, QtCore
import sys, re, logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class Property(QtCore.QObject):
    ''' Abstract Class to store properties in a QObject tree and connect to an editor
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    extended : property
               Extended Python property
    parent : QObject
             Parent object
    '''
    
    PROPERTIES = []
    
    def __init__(self, name, group, property=None, extended=None, parent=None):
        "Initialize a Property"
        
        QtCore.QObject.__init__(self, parent)
        self.setObjectName(name)
        self.property_obj = property
        self.group = group
        #self.property_ext = extended
        assert(property is None or hasattr(property, "dynamicPropertyNames"))
        self.hints = extended.editorHints if extended is not None else {}
        self.doc = extended.doc if extended is not None else None
        
        if 'label' in self.hints:
            self.displayName = self.hints['label']
        else:
            name = name[0].capitalize()+name[1:]
            if name.find('_') == -1:
                vals = re.findall('[A-Z][a-z]*', name)
            else:
                vals = name.replace('_', ' ').split()
            self.displayName = " ".join([v.capitalize() for v in vals]) if len(vals) > 0 else name.capitalize()
    
    def createEditor(self, parent, option):
        '''Returns the widget used to edit the item for editing. The parent 
        widget and style option are used to control how the editor widget appears. (abstract)
        
        :Parameters:
    
        parent : QWidget
                 Parent of created widget
        option : QStyleOptionViewItem
                 Used to describe the parameters used to draw an item in a view widget
        
        :Returns:
        
        val : QWidget
              Widget to edit the cell value
        '''
        
        pass
    
    def isBool(self):
        '''Test if property defines a Bool
        
        :Returns:
        
        val : bool
              False
        '''
        
        return False
    
    def setEditorData(self, editor, data):
        '''Sets the data to be displayed and edited by the editor from 
        the data model item. (does nothing)
        
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        data : QVariant
                Data to set in the editor
                
        :Returns:
        
        val : bool
              False
        '''
        
        return False
    
    def editorData(self, editor):
        ''' Get the data from a finished editor.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        
        :Returns:
        
        val : QVariant
              Data from the editor (empty)
        '''
        
        return QtCore.QVariant()
    
    def value(self, role = QtCore.Qt.UserRole):
        ''' Get the value for the given role
        
        :Parameters:
        
        role : enum
               Stored value role (Unused)
        
        :Returns:
        
        value : QVariant
                Stored value
        '''
        
        if self.property_obj is not None:
            return self.property_obj.property(self.objectName())
        return QtCore.QVariant()
    
    def __cmp__(self, other):
        ''' Compare two properties
        
        :Parameters:
        
        other : object
                String or Property to compare
        
        :Returns:
        
        val : bool
              True if both names are equals
        '''
        
        if isinstance(other, Property):
            return cmp(self.objectName(), other.objectName())
        if isinstance(other, str):
            return cmp(self.objectName(), other)
        return False
    
    @QtCore.pyqtSlot(int)
    @QtCore.pyqtSlot('float')
    @QtCore.pyqtSlot('double')
    def setValue(self, value):
        '''Set the value for the property
        
        :Parameters:
        
        value : QObject
               Value to store
        '''
        
        if self.property_obj is not None:
            return self.property_obj.setProperty(self.objectName(), value)
    
    def isReadOnly(self):
        ''' Test if the property is Read Only
        
        :Returns:
        
        val : bool
              False if property is writable
        '''
        
        if self.property_obj is not None:
            if self.property_obj.dynamicPropertyNames().count(self.objectName().toLocal8Bit()) > 0: return False
            prop = self.property_obj.metaObject().property(self.property_obj.metaObject().indexOfProperty(self.objectName()))
            if prop.isWritable() and not prop.isConstant(): return False
        return True
    
    def setEditorHints(self, hints):
        ''' Set the Editor Hints
        
        :Parameters:
        
        hints : str
                Editor hints
        '''
        
        self.hints = hints
        
    def editorHints(self):
        ''' Get editor hints
        
        :Returns:
        
        val : QString
              Editor Hints
        '''
        
        return self.hints
    
    def property_object(self):
        ''' Get reference to property object
        
        :Returns:
        
        val : QObject
              Property object
        '''
        
        return self.property_obj
    
    def isRoot(self):
        ''' Test if Property is root
        
        :Returns:
        
        val : bool
              True if no property is referenced
        '''
        
        return self.property_obj is None
    
    def row(self):
        ''' Get the row of the property in the greater list
        
        :Returns:
        
        row : int
              Row offset in property tree
        '''
        
        return self.parent().children().index(self)
    
    def findPropertyObject(self, property):
        ''' Recursively search for specified property
        
        :Parameters:
    
        property : QObject
                    Property object to find
        
        :Returns:
        
        val : Property
              Property encapsulation class containing the property
        '''
        
        if self.property_obj == property: return self
        for child in self.children():
            obj = child.findPropertyObject(property)
            if obj is not None: return obj
        return None
    
    def totalChildren(self):
        ''' Count the total number of children under this node.
        
        :Returns:
        
        total : int
                Total number of children
        '''
        
        total = 0
        for c in self.children():
            total += c.totalChildren()
        return total+1
    
    def maximumNameWidth(self, fontMetric, indent, depth=1):
        '''Get the maximum width of the property name using the given Font Metrics
        
        :Parameters:
        
        fontMetric : QFontMetrics
                      Metric to measure font size
        indent : int
                 Width of indent
        depth : int
                Depth of node
        
        :Returns:
        
        width : int
                Maximum width of text given the font
        '''
        
        width = fontMetric.width(self.displayName) + indent*depth
        for c in self.children():
            w = c.maximumNameWidth(fontMetric, indent, depth+1)
            if w > width: width = w
        return width
    
    def saveState(self, settings):
        ''' Save the state of the properties in the tree
        
        :Parameters:
        
        settings : QSettings
                   Save settings to platform specific location
        '''
        
        if self.property_obj is not None:
            _logger.debug("Save value %s = %s"%(self.objectName(), str(self.value())))
            settings.setValue(self.objectName(), self.value(QtCore.Qt.EditRole))
        if len(self.children()) > 0: 
            _logger.debug("Save Group %s with %d"%(self.objectName(), len(self.children())))
            settings.beginGroup(self.objectName())
            for child in self.children():
                child.saveState(settings)
            settings.endGroup()
    
    def restoreState(self, settings):
        ''' Restore the state of the properties in the tree
        
        :Parameters:
        
        settings : QSettings
                   Load settings from platform specific location
        '''
        
        if self.property_obj is not None:
            val = settings.value(self.objectName(), self.value(QtCore.Qt.EditRole))
            self.setValue(val)
            _logger.debug("Restore value %s = %s"%(self.objectName(), str(val)))
        if len(self.children()) > 0: 
            _logger.debug("Restore Group %s with %d"%(self.objectName(), len(self.children())))
            settings.beginGroup(self.objectName())
            for child in self.children():
                child.restoreState(settings)
            settings.endGroup()

def register_property(name, bases, dict):
    ''' Register a property subclass with Property
    
    :Parameters:
    
    name : str
           Name of the class
    bases : list
            Base classes
    dict : dict
           Class attributes
    
    :Returns:
    
    type : type
           Type of the class
    '''
    
    classType = type(name, bases, dict)
    Property.PROPERTIES.append(classType)
    return classType

class ChoiceProperty(Property):
    '''Connect a choice property to a QComboBox
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    extended : property
              Extended Python property
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, extended=None, parent=None):
        "Initialize a Choice Property"
        
        Property.__init__(self, name, group, property, extended, parent)
        self.choices = extended.editorHints["choices"]
        self.use_int = property.property(name).type() == QtCore.QVariant.Int
    
    @classmethod
    def create(cls, name, group, property=None, extended=None, parent=None):
        ''' Test if property holds a numeric type and if so return a ChoiceProperty
        
        :Parameters:
        
        name : str
               Name of the property
        group : int
                Group index
        property : QObject
                   Property object
        extended : property
                  Extended Python property
        parent : QObject
               Parent object
        
        :Returns:
        
        val : BoolProperty
              Property object
        '''
        
        _logger.debug("Create ChoiceProperty: %s - %s - %s"%(name, str(property.property(name).type()), str(extended.editorHints)))
        if (property.property(name).type() == QtCore.QVariant.Int or property.property(name).type() == QtCore.QVariant.String) and "choices" in extended.editorHints:
            return cls(name, group, property, extended, parent)
        return None
    
    def createEditor(self, parent, option):
        '''Returns the widget used to edit the item for editing. The parent 
        widget and style option are used to control how the editor widget appears.
        
        :Parameters:
    
        parent : QWidget
                 Parent of created widget
        option : QStyleOptionViewItem
                 Used to describe the parameters used to draw an item in a view widget
        
        :Returns:
        
        val : QWidget
              Widget to edit the cell value
        '''
        
        _logger.debug("Create QComboBox")
        editor = QtGui.QComboBox(parent)
        choices = self.choices(self.property_obj) if callable(self.choices) else self.choices
        for name in choices:
            editor.addItem(name.replace('_', ' '))
        self.connect(editor, QtCore.SIGNAL("currentIndexChanged(int)"), self, QtCore.SLOT("setValue(int)"))
        _logger.debug("Create QComboBox - finished")
        return editor    
    
    def setValue(self, value):
        '''Set the value for the property
        
        :Parameters:
        
        value : QObject
               Value to store
        '''
        
        if isinstance(value, QtCore.QVariant):
            if value.type() == QtCore.QVariant.String:
                value = value.toString()
            elif value.type() == QtCore.QVariant.Int:
                val, check = value.toInt()
                if check: value = val

        if is_int(value):
            if self.use_int:
                Property.setValue(self, value)
            else:
                choices = self.choices(self.property_obj) if callable(self.choices) else self.choices
                Property.setValue(self, choices[value])
        else:
            try: value+"ddd"
            except:
                if isinstance(value, QtCore.QVariant):
                    _logger.warn("QVariant not supported - %s"%(str(value.type())))
                else:
                    _logger.warn("Value not supported - %s"%(str(value.__class__.__name__)))
            else:
                choices = self.choices(self.property_obj) if callable(self.choices) else self.choices
                try:
                    index = choices.index(value)
                except:
                    value = value.replace(' ', '_')
                try:index = choices.index(value)
                except: 
                    _logger.warn("Cannot find - %s in %s"%(str(value), str(self.choices)))
                else:
                    if self.use_int:
                        Property.setValue(self, index)
                    else:
                        Property.setValue(self, choices[index])
    
    def setEditorData(self, editor, data):
        '''Sets the data to be displayed and edited by the editor from 
        the data model item.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        data : QVariant
                Data to set in the editor
        
        :Returns:
        
        val : bool
              True if new value was set
        '''
        
        if isinstance(data, QtCore.QVariant): data = data.toString()
        index = editor.findText(data)
        if index == -1: return False
        editor.blockSignals(True)
        editor.setCurrentIndex(index)
        editor.blockSignals(False)
        return True
    
    def editorData(self, editor):
        ''' Get the data from a finished editor.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        
        :Returns:
        
        val : QVariant
              Data from the editor
        '''
        
        #return QtCore.QVariant(editor.currentIndex())
        return QtCore.QVariant(editor.currentText())
    
    def value(self, role = QtCore.Qt.UserRole):
        ''' Get the value for the given role
        
        :Parameters:
        
        role : enum
               Stored value role (Unused)
        
        :Returns:
        
        value : QVariant
                Stored value
        '''
        
        if self.property_obj is not None:
            val = self.property_obj.property(self.objectName())
            index, check = val.toInt()
            if check:
                choices = self.choices(self.property_obj) if callable(self.choices) else self.choices
                try:
                    return QtCore.QVariant(choices[index].replace('_', ' '))
                except:
                    _logger.exception("Index out of bounds %d > %d -> %s -- %s"%(index, len(choices), str(choices), str(self.displayName)))
                    return QtCore.QVariant()
            elif val.type() == QtCore.QVariant.String:
                return val
            else:
                _logger.debug("Value type not supported as an index - %s"%(str(val.type())))
        return QtCore.QVariant()

class NumericProperty(Property):
    '''Connect a Numeric property to a QSpinBox
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    extended : property
              Extended Python property
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    NUMERIC_TYPES = (QtCore.QVariant.Int, QtCore.QMetaType.Float, QtCore.QVariant.Double)
    
    def __init__(self, name, group, property=None, extended=None, parent=None):
        "Initialize a Numeric Property"
        
        Property.__init__(self, name, group, property, extended, parent)
        #editorHints
        self.minimum = extended.editorHints["minimum"] if hasattr(extended, 'editorHints') and "minimum" in extended.editorHints else -sys.maxint
        self.maximum = extended.editorHints["maximum"] if hasattr(extended, 'editorHints') and "maximum" in extended.editorHints else sys.maxint
        _logger.debug("NumericProperty::minimum %d, %s, %s"%(hasattr(extended, 'editorHints'), name, str(self.minimum)))
        _logger.debug("NumericProperty::minimum %d, %s, %s"%(hasattr(extended, 'editorHints'), name, str(self.maximum)))
        if self.value().type() == QtCore.QVariant.Int:
            self.singleStep = extended.editorHints["singleStep"] if hasattr(extended, 'editorHints') and "singleStep" in extended.editorHints else 1
        else:
            self.singleStep = extended.editorHints["singleStep"] if hasattr(extended, 'editorHints') and "singleStep" in extended.editorHints else 0.1
            self.decimals = extended.editorHints["decimals"] if hasattr(extended, 'editorHints') and "decimals" in extended.editorHints else 2
    
    @classmethod
    def create(cls, name, group, property=None, extended=None, parent=None):
        ''' Test if property holds a numeric type and if so return a NumericProperty
        
        :Parameters:
        
        name : str
               Name of the property
        group : int
                Group index
        property : QObject
                   Property object
        extended : property
                  Extended Python property
        parent : QObject
               Parent object
        
        :Returns:
        
        val : NumericProperty
              Property object
        '''
        
        _logger.debug("Create NumericProperty: %s - %s - %s"%(name, str(property.property(name).type()), str(extended.editorHints)))
        if property.property(name).type() in NumericProperty.NUMERIC_TYPES:# and "minimum" in extended.editorHints:
            return cls(name, group, property, extended, parent)
        return None
    
    def createEditor(self, parent, option):
        '''Returns the widget used to edit the item for editing. The parent 
        widget and style option are used to control how the editor widget appears.
        
        :Parameters:
    
        parent : QWidget
                 Parent of created widget
        option : QStyleOptionViewItem
                 Used to describe the parameters used to draw an item in a view widget
        
        :Returns:
        
        val : QWidget
              Widget to edit the cell value
        '''
        
        editor = None
        type = self.value().type()
        minimum = self.minimum(self.property_obj) if callable(self.minimum) else self.minimum
        maximum = self.maximum(self.property_obj) if callable(self.maximum) else self.maximum
        singleStep = self.singleStep(self.property_obj) if callable(self.singleStep) else self.singleStep
        _logger.debug("SpinBox(%d,%d,%d)"%(minimum, maximum, singleStep))
        if type == QtCore.QVariant.Int:
            editor = QtGui.QSpinBox(parent)
            editor.setProperty("minimum", minimum)
            editor.setProperty("maximum",  maximum)
            editor.setProperty("singleStep",  singleStep)
            self.connect(editor, QtCore.SIGNAL("valueChanged(int)"), self, QtCore.SLOT("setValue(int)"))
        elif type == QtCore.QMetaType.Float or type == QtCore.QVariant.Double:
            decimals = self.decimals(self.property_obj) if callable(self.decimals) else self.decimals
            editor = QtGui.QDoubleSpinBox(parent)
            editor.setProperty("minimum", minimum)
            editor.setProperty("maximum",  maximum)
            editor.setProperty("singleStep",  singleStep)
            editor.setProperty("decimals",  decimals)
            self.connect(editor, QtCore.SIGNAL("valueChanged(double)"), self, QtCore.SLOT("setValue(double)"))
        return editor
    
    def setEditorData(self, editor, data):
        '''Sets the data to be displayed and edited by the editor from 
        the data model item.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        data : QVariant
                Data to set in the editor
        
        :Returns:
        
        val : bool
              True if new value was set
        '''
        
        type = self.value().type()
        if type == QtCore.QVariant.Int:
            editor.blockSignals(True)
            val, check = data.toInt()
            if check: editor.setValue(val)
            editor.blockSignals(False)
            return check
        elif type == QtCore.QMetaType.Float or type == QtCore.QVariant.Double:
            editor.blockSignals(True)
            val, check = data.toDouble()
            if check: editor.setValue(val)
            editor.blockSignals(False)
            return check
        return False
    
    def editorData(self, editor):
        ''' Get the data from a finished editor.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        
        :Returns:
        
        val : QVariant
              Data from the editor
        '''
        
        type = self.value().type()
        if type in NumericProperty.NUMERIC_TYPES:
            return QtCore.QVariant(editor.value())
        return QtCore.QVariant()

class BoolProperty(Property):
    '''Connect a bool property to a QCheckBox
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    extended : property
              Extended Python property
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, extended=None, parent=None):
        "Initialize a Boolean Property"
        
        Property.__init__(self, name, group, property, extended, parent)
    
    @classmethod
    def create(cls, name, group, property=None, extended=None, parent=None):
        ''' Test if property holds a numeric type and if so return a BoolProperty
        
        :Parameters:
        
        name : str
               Name of the property
        group : int
                Group index
        property : QObject
                   Property object
        extended : property
                  Extended Python property
        parent : QObject
                 Parent object
        
        :Returns:
        
        val : BoolProperty
              Property object
        '''
        
        _logger.debug("Create BoolProperty: %s - %s"%(name, str(property.property(name).type()), ))
        if property.property(name).type() == QtCore.QVariant.Bool:
            return cls(name, group, property, extended, parent)
        return None
    
    def isBool(self):
        '''Test if property defines a Bool
        
        :Returns:
        
        val : bool
              True
        '''
        
        return True
    
    def value(self, role = QtCore.Qt.UserRole):
        ''' Get the value for the given role
        
        :Parameters:
        
        role : enum
               Stored value role (Unused)
        
        :Returns:
        
        value : QVariant
                Stored value
        '''
        
        
        if role == QtCore.Qt.CheckStateRole:
            val = self.property_obj.property(self.objectName())
            if isinstance(val, QtCore.QVariant): val = val.toBool()
            return QtCore.Qt.Checked if val else QtCore.Qt.Unchecked
        if role == QtCore.Qt.DisplayRole: return QtCore.QVariant("")
        if self.property_obj is not None:
            return self.property_obj.property(self.objectName())
        return QtCore.QVariant()
    
    def createEditor(self, parent, option):
        '''Returns the widget used to edit the item for editing. The parent 
        widget and style option are used to control how the editor widget appears.
        
        :Parameters:
    
        parent : QWidget
                 Parent of created widget
        option : QStyleOptionViewItem
                 Used to describe the parameters used to draw an item in a view widget
        
        :Returns:
        
        val : QWidget
              Widget to edit the cell value
        '''
        
        editor = QtGui.QCheckBox(parent) #CheckboxWidget(parent)
        editor.setFocusPolicy(QtCore.Qt.StrongFocus)
        #self.connect(editor.button, QtCore.SIGNAL("stateChanged(int)"), self, QtCore.SLOT("setValue(int)"))
        self.connect(editor, QtCore.SIGNAL("stateChanged(int)"), self, QtCore.SLOT("setValue(int)"))
        return editor
    
    def setEditorData(self, editor, data):
        '''Sets the data to be displayed and edited by the editor from 
        the data model item.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        data : QVariant
                Data to set in the editor
        
        :Returns:
        
        val : bool
              True if new value was set
        '''
        
        #editor = editor.button
        editor.blockSignals(True)
        if data.toBool(): editor.setCheckState(QtCore.Qt.Checked)
        else: editor.setCheckState(QtCore.Qt.Unchecked)
        editor.blockSignals(False)
        return True
    
    def editorData(self, editor):
        ''' Get the data from a finished editor.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        
        :Returns:
        
        val : QVariant
              Data from the editor
        '''
        
        #return QtCore.QVariant(editor.button.checkState() == QtCore.Qt.Checked)
        return QtCore.QVariant(editor.checkState() == QtCore.Qt.Checked)

class FontProperty(Property):
    '''Connect a font property to a QFontDialog
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    extended : property
               Extended Python property
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, extended=None, parent=None):
        "Initialize a Choice Property"
        
        Property.__init__(self, name, group, property, extended, parent)
    
    @classmethod
    def create(cls, name, group, property=None, extended=None, parent=None):
        ''' Test if property holds a numeric type and if so return a ChoiceProperty
        
        :Parameters:
        
        name : str
               Name of the property
        group : int
                Group index
        property : QObject
                   Property object
        extended : property
                  Extended Python property
        parent : QObject
               Parent object
        
        :Returns:
        
        val : BoolProperty
              Property object
        '''
        
        _logger.debug("Create FontProperty: %s - %s"%(name, str(property.property(name).type())))
        if property.property(name).type() == QtCore.QVariant.Font:
            return cls(name, group, property, extended, parent)
        return None

    def createEditor(self, parent, option):
        '''Returns the widget used to edit the item for editing. The parent 
        widget and style option are used to control how the editor widget appears.
        
        :Parameters:
    
        parent : QWidget
                 Parent of created widget
        option : QStyleOptionViewItem
                 Used to describe the parameters used to draw an item in a view widget
        
        :Returns:
        
        val : QWidget
              Widget to edit the cell value
        '''
        
        _logger.debug("Create QFontDialog")
        editor = FontDialogWidget(parent)
        self.connect(editor, QtCore.SIGNAL("fontChanged(const QFont&)"), self, QtCore.SLOT("setValue(const QFont&)"))
        _logger.debug("Create QComboBox - finished")
        return editor
    
    def setEditorData(self, editor, data):
        '''Sets the data to be displayed and edited by the editor from 
        the data model item.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        data : QVariant
                Data to set in the editor
        
        :Returns:
        
        val : bool
              True if new value was set
        '''
        
        editor.setCurrentFont(QtGui.QFont(data))
        return True
    
    def editorData(self, editor):
        ''' Get the data from a finished editor.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        
        :Returns:
        
        val : QVariant
              Data from the editor
        '''
        
        #return QtCore.QVariant(editor.currentIndex())
        return QtCore.QVariant(editor.selectedFont())
    
    @QtCore.pyqtSlot('const QFont&')
    def setValue(self, value):
        '''Set the value for the property
        
        :Parameters:
        
        value : QObject
               Value to store
        '''
        
        value = QtGui.QFont(value)
        Property.setValue(self, value)
    
    def value(self, role = QtCore.Qt.UserRole):
        ''' Get the value for the given role
        
        :Parameters:
        
        role : enum
               Stored value role (Unused)
        
        :Returns:
        
        value : QVariant
                Stored value
        '''
        
        if self.property_obj is not None:
            val = QtGui.QFont(self.property_obj.property(self.objectName()))
            if role == QtCore.Qt.FontRole or role == QtCore.Qt.EditRole: return val
            return QtCore.QVariant(val.family()+" (%d)"%val.pointSize())
        return QtCore.QVariant()

class FilenameProperty(Property):
    '''Connect a font property to a file dialog
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    extended : property
               Extended Python property
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, extended=None, parent=None):
        "Initialize a Choice Property"
        
        Property.__init__(self, name, group, property, extended, parent)
        self.filter = extended.editorHints["filter"] if 'filter' in extended.editorHints else ""
        self.path = extended.editorHints["path"]if 'path' in extended.editorHints else ""
        self.filetype = extended.editorHints["filetype"]
    
    @classmethod
    def create(cls, name, group, property=None, extended=None, parent=None):
        ''' Test if property holds a numeric type and if so return a ChoiceProperty
        
        :Parameters:
        
        name : str
               Name of the property
        group : int
                Group index
        property : QObject
                   Property object
        extended : property
                  Extended Python property
        parent : QObject
               Parent object
        
        :Returns:
        
        val : BoolProperty
              Property object
        '''
        
        _logger.debug("Create FilenameProperty: %s - %s - %s"%(name, str(property.property(name).type()), str(extended.editorHints)))
        if property.property(name).type() == QtCore.QVariant.String and 'filetype' in extended.editorHints:
            return cls(name, group, property, extended, parent)
        return None

    def createEditor(self, parent, option):
        '''Returns the widget used to edit the item for editing. The parent 
        widget and style option are used to control how the editor widget appears.
        
        :Parameters:
    
        parent : QWidget
                 Parent of created widget
        option : QStyleOptionViewItem
                 Used to describe the parameters used to draw an item in a view widget
        
        :Returns:
        
        val : QWidget
              Widget to edit the cell value
        '''
        
        editor = FileDialogWidget(self.filetype, self.filter, self.path, parent)
        self.connect(editor, QtCore.SIGNAL("fileChanged(const QString&)"), self, QtCore.SLOT("setValue(const QString&)"))
        return editor
    
    def setEditorData(self, editor, data):
        '''Sets the data to be displayed and edited by the editor from 
        the data model item.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        data : QVariant
                Data to set in the editor
        
        :Returns:
        
        val : bool
              True if new value was set
        '''
        
        _logger.debug("FilenameProperty type %s"%(data.__class__))
        if data.type() == QtCore.QVariant.String:
            editor.setCurrentFilename(data.toString())
            return True
        else:
            return False
    
    def editorData(self, editor):
        ''' Get the data from a finished editor.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        
        :Returns:
        
        val : QVariant
              Data from the editor
        '''
        
        #return QtCore.QVariant(editor.currentIndex())
        return QtCore.QVariant(editor.selectedFilename())
    
    @QtCore.pyqtSlot('const QString&')
    def setValue(self, value):
        '''Set the value for the property
        
        :Parameters:
        
        value : QObject
               Value to store
        '''
        
        _logger.debug("setValue Qstring")
        if not hasattr(value, 'isValid'):
            Property.setValue(self, str(value))
        elif value.isValid():
            if value.type() == QtCore.QVariant.String:
                Property.setValue(self, str(value.toString()))
            elif value.type() == QtCore.QVariant.StringList:
                value = [str(s) for s in value]
                Property.setValue(self, value)
            else:
                raise ValueError, "bug: %s -- %"%(str(value.typeName()), str(value))
    
    def value(self, role = QtCore.Qt.UserRole):
        ''' Get the value for the given role
        
        :Parameters:
        
        role : enum
               Stored value role (Unused)
        
        :Returns:
        
        value : QVariant
                Stored value
        '''
        
        return Property.value(self, role)

class WorkflowProperty(Property):
    '''Connect a font property to a Workflow dialog
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    extended : property
               Extended Python property
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, extended=None, parent=None):
        "Initialize a Choice Property"
        
        Property.__init__(self, name, group, property, extended, parent)
        self.operations = extended.editorHints["operations"]
    
    @classmethod
    def create(cls, name, group, property=None, extended=None, parent=None):
        ''' Test if property holds a numeric type and if so return a ChoiceProperty
        
        :Parameters:
        
        name : str
               Name of the property
        group : int
                Group index
        property : QObject
                   Property object
        extended : property
                  Extended Python property
        parent : QObject
               Parent object
        
        :Returns:
        
        val : BoolProperty
              Property object
        '''
        
        _logger.debug("Create WorkflowProperty: %s - %s - %s"%(name, str(property.property(name).type()), str(extended.editorHints)))
        if property.property(name).type() == QtCore.QVariant.String and 'operations' in extended.editorHints:
            return cls(name, group, property, extended, parent)
        return None

    def createEditor(self, parent, option):
        '''Returns the widget used to edit the item for editing. The parent 
        widget and style option are used to control how the editor widget appears.
        
        :Parameters:
    
        parent : QWidget
                 Parent of created widget
        option : QStyleOptionViewItem
                 Used to describe the parameters used to draw an item in a view widget
        
        :Returns:
        
        val : QWidget
              Widget to edit the cell value
        '''
        
        editor = WorkflowWidget(self.operations, parent)
        self.connect(editor, QtCore.SIGNAL("operationsUpdated(PyQt_PyObject)"), self.setValue) #, QtCore.SLOT("setValue(PyQt_PyObject)"))
        return editor
    
    def setEditorData(self, editor, data):
        '''Sets the data to be displayed and edited by the editor from 
        the data model item.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        data : QVariant
                Data to set in the editor
        
        :Returns:
        
        val : bool
              True if new value was set
        '''
        
        _logger.debug("FilenameProperty type %s"%(data.__class__))
        if data.type() == QtCore.QVariant.String:
            editor.setWorkflow(str(data.toString()).split(','))
            return True
        else:
            return False
    
    def editorData(self, editor):
        ''' Get the data from a finished editor.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        
        :Returns:
        
        val : QVariant
              Data from the editor
        '''
        
        #return QtCore.QVariant(editor.currentIndex())
        print "editor=", editor
        return QtCore.QVariant(",".join(editor.workflow()))
    
    @QtCore.pyqtSlot('PyQt_PyObject')
    def setValue(self, value):
        '''Set the value for the property
        
        :Parameters:
        
        value : QObject
               Value to store
        '''
        
        if isinstance(value, list):
            value = ",".join(value)
        _logger.debug("setValue PyQt_PyObject")
        Property.setValue(self, value)
    
    def value(self, role = QtCore.Qt.UserRole):
        ''' Get the value for the given role
        
        :Parameters:
        
        role : enum
               Stored value role (Unused)
        
        :Returns:
        
        value : QVariant
                Stored value
        '''
        
        return Property.value(self, role)

class StringProperty(Property):
    '''Connect a String property to a QLineEdit
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    extended : property
              Extended Python property
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, extended=None, parent=None):
        "Initialize a String Property"
        
        Property.__init__(self, name, group, property, extended, parent)
    
    @classmethod
    def create(cls, name, group, property=None, extended=None, parent=None):
        ''' Test if property holds a string type and if so return a StringProperty
        
        :Parameters:
        
        name : str
               Name of the property
        group : int
                Group index
        property : QObject
                   Property object
        extended : property
                  Extended Python property
        parent : QObject
               Parent object
        
        :Returns:
        
        val : StringProperty
              Property object
        '''
        
        _logger.debug("Create StringProperty: %s - %s"%(name, str(property.property(name).type())))
        if property.property(name).type() == QtCore.QVariant.String:
            return cls(name, group, property, extended, parent)
        return None
    
def is_int(f):
    '''Test if the float value is an integer
    
    This function casts the float to an integer and subtracts it from the float
        - if the result is zero, then return True
        - otherwise, return False
    
    .. sourcecode:: py
    
        >>> from core.metadata.type_utility import *
        >>> is_float_int(1.0)
        True
        >>> is_float_int(1.1)
        False
    
    :Parameters:

    obj : float
          A float value
        
    :Returns:
        
    return_val : boolean
                 True if float holds an integer
    '''
    
    try:
        f = float(f)
        i = int(f)
        if (f-i) == 0: return True
        else: return False
    except:
        return False

