'''
This class defines a tree model.

Adopted from http://qt-apps.org/content/show.php/QPropertyEditor?content=68684
Original Author: Volker Wiendl with Enhancements by Roman alias banal

.. Created on Dec 2, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ButtonDelegate import FontDialogWidget, FileDialogWidget #, WorkflowWidget #, CheckboxWidget
from ..util.qt4_loader import QtGui, QtCore, qtSlot, qtSignal 
import re, logging, glob, os

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
    hints : dict
            GUI hints
    doc : str
          GUI help string
    parent : QObject
             Parent object
    '''
    
    PROPERTIES = []
    
    propertyValidity = qtSignal(object, bool)
    
    def __init__(self, name, group=0, property=None, hints={}, doc={}, parent=None):
        "Initialize a Property"
        
        QtCore.QObject.__init__(self, parent)
        self.setObjectName(name)
        self.property_obj = property
        self.group = group
        self.doc = doc
        self.hints = hints
        self.required = 'required' in hints and hints['required']
        
        if 'label' in self.hints:
            self.displayName = self.hints['label']
        else:
            name = name[0].capitalize()+name[1:]
            if name.find('_') == -1:
                vals = re.findall('[A-Z][a-z]*', name)
            else:
                vals = name.replace('_', ' ').split()
            self.displayName = " ".join([v.capitalize() for v in vals]) if len(vals) > 0 else name.capitalize()
    
    def isValid(self):
        ''' Test if the property holds a valid value
        
        :Returns:
        
        val : bool
              True by default for most properties
        '''
        
        return True
    
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
        data : object
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
        
        val : object
              Data from the editor (empty)
        '''
        
        return None
    
    def value(self, role = QtCore.Qt.UserRole):
        ''' Get the value for the given role
        
        :Parameters:
        
        role : enum
               Stored value role (Unused)
        
        :Returns:
        
        value : object
                Stored value
        '''
        
        if self.property_obj is not None:
            return self.property_obj.property(self.objectName())
        return None
    
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
    
    @qtSlot(int)
    @qtSlot('float')
    @qtSlot('double')
    def setValue(self, value, valid=True):
        '''Set the value for the property
        
        :Parameters:
        
        value : QObject
               Value to store
        '''
        
        if self.property_obj is not None:
            #if self.required: 
            if self.property_obj.property(self.objectName()) != value:
                self.propertyValidity.emit(self, valid)
            return self.property_obj.setProperty(self.objectName(), value)
    
    def isReadOnly(self):
        ''' Test if the property is Read Only
        
        :Returns:
        
        val : bool
              False if property is writable
        '''
        
        if self.property_obj is not None:
            if hasattr(self.property_obj, 'dynamicPropertyNames') and self.property_obj.dynamicPropertyNames().count(self.objectName()) > 0: return False
            if hasattr(self.property_obj, 'metaObject'):
                prop = self.property_obj.metaObject().property(self.property_obj.metaObject().indexOfProperty(self.objectName()))
                if prop.isWritable() and not prop.isConstant(): return False
            else: 
                if 'readonly' in self.hints and self.hints['readonly']: return True
                return False
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
        
        val : str
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
    
    def totalInvalid(self):
        ''' Count the total number of children under this node.
        
        :Returns:
        
        total : int
                Total number of invalid, but required children
        '''
        
        total = 0
        for c in self.children():
            total += c.totalInvalid()
        if self.required and not self.isValid(): 
            total += 1
        return total
    
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

def parseHints(name, obj):
    ''' Get editor hints from documentation
    
    :Parameters:
    
    obj : object
          Input property object, 
    
    :Returns:
    
    hints : dict
            Editor hint map
    '''
    
    hints={}
    doc = ""
    if obj is not None:
        if hasattr(obj, 'doc'): doc = obj.doc
        if hasattr(obj, 'editorHints'): return name, obj.editorHints, doc
        doc = obj if isinstance(obj, str) else obj.__doc__
        if doc is not None:
            beg = doc.find('{')
            end = doc.find('}')
            if beg != -1 and end != -1:
                return name, dict(doc[beg:end]), doc
    else:
        obj = name
        name = obj.dest
        doc = obj.help
        if hasattr(obj, 'gui_hint') and obj.gui_hint is not None:
            hints = obj.gui_hint
        
    return name, hints, doc

class ChoiceProperty(Property):
    '''Connect a choice property to a QComboBox
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    hints : dict
            GUI hints
    doc : str
          GUI help string
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, hints={}, doc="", parent=None):
        "Initialize a Choice Property"
        
        Property.__init__(self, name, group, property, hints, doc, parent)
        self.choices = self.hints["choices"]
        self.use_int = isinstance( property.property(name), ( int, long ) )
    
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
        extended : object
                   Additional meta information
        parent : QObject
               Parent object
        
        :Returns:
        
        val : BoolProperty
              Property object
        '''
        
        name, hints, doc = parseHints(name, extended)
        _logger.debug("Create ChoiceProperty: %s - %s - %s"%(name, str(property.property(name).__class__), str(hints)))
        val = property.property(name)
        if( isinstance( val, ( int, long ) ) or isinstance( val, basestring )  ) and "choices" in hints:
            return cls(name, group, property, hints, doc, parent)
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

        if is_int(value):
            if self.use_int:
                Property.setValue(self, value)
            else:
                choices = self.choices(self.property_obj) if callable(self.choices) else self.choices
                Property.setValue(self, choices[value])
        else:
            try: value+"ddd"
            except:
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
        data : object
                Data to set in the editor
        
        :Returns:
        
        val : bool
              True if new value was set
        '''
        
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
        
        val : object
              Data from the editor
        '''
        
        #return editor.currentIndex()
        return editor.currentText()
    
    def value(self, role = QtCore.Qt.UserRole):
        ''' Get the value for the given role
        
        :Parameters:
        
        role : enum
               Stored value role (Unused)
        
        :Returns:
        
        value : object
                Stored value
        '''
        
        if self.property_obj is not None:
            val = self.property_obj.property(self.objectName())
            try:
                index = int(val)
            except:
                if isinstance( val, basestring ):
                    return val
                else:
                    _logger.debug("Value type not supported as an index - %s"%(str(val.__class__)))
            else:
                choices = self.choices(self.property_obj) if callable(self.choices) else self.choices
                try:
                    return choices[index].replace('_', ' ')
                except:
                    _logger.exception("Index out of bounds %d > %d -> %s -- %s"%(index, len(choices), str(choices), str(self.displayName)))
                    return None
        return None

class NumericProperty(Property):
    '''Connect a Numeric property to a QSpinBox
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    hints : dict
            GUI hints
    doc : str
          GUI help string
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, hints={}, doc="", parent=None):
        "Initialize a Numeric Property"
        
        Property.__init__(self, name, group, property, hints, doc, parent)
        #editorHints
        self.minimum = self.hints["minimum"] if "minimum" in self.hints else -32767
        self.maximum = self.hints["maximum"] if "maximum" in self.hints else 32767
        
        _logger.debug("NumericProperty::minimum %s, %s"%(name, str(self.minimum)))
        _logger.debug("NumericProperty::minimum %s, %s"%(name, str(self.maximum)))
        if isinstance( self.value(), ( int, long ) ):
            self.singleStep = self.hints["singleStep"] if "singleStep" in self.hints else 1
        else:
            self.singleStep = self.hints["singleStep"] if "singleStep" in self.hints else 0.1
            self.decimals = self.hints["decimals"] if "decimals" in self.hints else 2
    
    def setValue(self, val):
        '''
        '''
        
        if isinstance( self.value(), ( int, long ) ):
            val = int(val)
        else:
            val = float(val)
        Property.setValue(self, val)
    
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
        extended : object
                   Additional meta information
        parent : QObject
               Parent object
        
        :Returns:
        
        val : NumericProperty
              Property object
        '''
        
        name, hints,doc = parseHints(name, extended)
        _logger.debug("Create NumericProperty: %s - %s - %s | %d"%(name, str(property.property(name).__class__), str(hints), isinstance( property.property(name), ( int, long, float ) )))
        if isinstance( property.property(name), ( int, long, float ) ) and not isinstance(property.property(name), bool):
            return cls(name, group, property, hints, doc, parent)
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
        val = self.value()
        minimum = self.minimum(self.property_obj) if callable(self.minimum) else self.minimum
        maximum = self.maximum(self.property_obj) if callable(self.maximum) else self.maximum
        singleStep = self.singleStep(self.property_obj) if callable(self.singleStep) else self.singleStep
        _logger.debug("SpinBox(%d,%d,%d)"%(minimum, maximum, singleStep))
        
        if isinstance( val, ( int, long ) ):
            editor = QtGui.QSpinBox(parent)
            editor.setProperty("minimum", minimum)
            editor.setProperty("maximum",  maximum)
            editor.setProperty("singleStep",  singleStep)
            self.connect(editor, QtCore.SIGNAL("valueChanged(int)"), self, QtCore.SLOT("setValue(int)"))
        elif isinstance(val, float):
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
        data : object
                Data to set in the editor
        
        :Returns:
        
        val : bool
              True if new value was set
        '''
        
        val = self.value()
        
        if isinstance( val, ( int, long ) ):
            editor.blockSignals(True)
            editor.setValue(val)
            editor.blockSignals(False)
            return True
        elif isinstance(val, float):
            editor.blockSignals(True)
            editor.setValue(val)
            editor.blockSignals(False)
            return True
        return False
    
    def editorData(self, editor):
        ''' Get the data from a finished editor.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        
        :Returns:
        
        val : object
              Data from the editor
        '''
        
        if isinstance( self.value(), ( int, long, float ) ):
            return editor.value()
        return None

class BoolProperty(Property):
    '''Connect a bool property to a QCheckBox
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    hints : dict
            GUI hints
    doc : str
          GUI help string
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, hints={}, doc="", parent=None):
        "Initialize a Boolean Property"
        
        Property.__init__(self, name, group, property, hints, doc, parent)
    
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
        extended : object
                   Additional meta information
        parent : QObject
                 Parent object
        
        :Returns:
        
        val : BoolProperty
              Property object
        '''
        
        name, hints,doc = parseHints(name, extended)
        _logger.debug("Create BoolProperty: %s - %s"%(name, str(property.property(name).__class__), ))
        
        if isinstance(property.property(name), bool):
            return cls(name, group, property, hints, doc, parent)
        return None
    
    def setValue(self, val):
        '''
        '''
        
        try: ""+val
        except:
            Property.setValue(self, val)
        else:
            Property.setValue(self, val.lower()=='true')
    
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
        
        value : object
                Stored value
        '''
        
        
        if role == QtCore.Qt.CheckStateRole:
            val = self.property_obj.property(self.objectName())
            return QtCore.Qt.Checked if val else QtCore.Qt.Unchecked
        if role == QtCore.Qt.DisplayRole: return ""
        if self.property_obj is not None:
            return self.property_obj.property(self.objectName())
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
        data : object
                Data to set in the editor
        
        :Returns:
        
        val : bool
              True if new value was set
        '''
        
        #editor = editor.button
        editor.blockSignals(True)
        if data: editor.setCheckState(QtCore.Qt.Checked)
        else: editor.setCheckState(QtCore.Qt.Unchecked)
        editor.blockSignals(False)
        return True
    
    def editorData(self, editor):
        ''' Get the data from a finished editor.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        
        :Returns:
        
        val : object
              Data from the editor
        '''
        
        #return editor.button.checkState() == QtCore.Qt.Checked
        return editor.checkState() == QtCore.Qt.Checked

class FontProperty(Property):
    '''Connect a font property to a QFontDialog
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    hints : dict
            GUI hints
    doc : str
          GUI help string
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, hints={}, doc="", parent=None):
        "Initialize a Choice Property"
        
        Property.__init__(self, name, group, property, hints, doc, parent)
    
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
        extended : object
                   Additional meta information
        parent : QObject
               Parent object
        
        :Returns:
        
        val : BoolProperty
              Property object
        '''
        
        name, hints,doc = parseHints(name, extended)
        _logger.debug("Create FontProperty: %s - %s"%(name, str(property.property(name).__class__)))
        if isinstance(property.property(name), QtGui.QFont):
            return cls(name, group, property, hints, doc, parent)
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
        data : object
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
        
        val : object
              Data from the editor
        '''
        
        #return editor.currentIndex()
        return editor.selectedFont()
    
    @qtSlot('const QFont&')
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
        
        value : object
                Stored value
        '''
        
        if self.property_obj is not None:
            val = QtGui.QFont(self.property_obj.property(self.objectName()))
            if role == QtCore.Qt.FontRole or role == QtCore.Qt.EditRole: return val
            return val.family()+" (%d)"%val.pointSize()
        return None

class FilenameProperty(Property):
    '''Connect a font property to a file dialog
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    hints : dict
            GUI hints
    doc : str
          GUI help string
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, hints={}, doc="", parent=None):
        "Initialize a Choice Property"
        
        Property.__init__(self, name, group, property, hints, doc, parent)
        self.filter = self.hints["filter"] if 'filter' in self.hints else ""
        self.path = self.hints["path"]if 'path' in self.hints else ""
        self.filetype = self.hints["filetype"]
        self.classtype = property.property(name).__class__
        #print 'here1', name, self.classtype
        if isinstance(property.property(name), list) and self.filetype=='open':
            #print 'here2', name, self.classtype
            self.filetype = 'file-list'
    
    def isValid(self):
        ''' Test if the property holds a valid value
        
        :Returns:
        
        val : bool
              True by default for most properties
        '''
        
        if not self.required: return True
        if self.filetype == 'file-list':
            return len(self.value()) > 0
        return self.value() != ""
    
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
        extended : object
                   Additional meta information
        parent : QObject
               Parent object
        
        :Returns:
        
        val : BoolProperty
              Property object
        '''
        
        name, hints,doc = parseHints(name, extended)
        _logger.debug("Create FilenameProperty: %s - %s - %s"%(name, str(property.property(name).__class__), str(hints)))
        if (isinstance(property.property(name), basestring) or isinstance(property.property(name), list)) and 'filetype' in hints:
            return cls(name, group, property, hints, doc, parent)
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
        editor.fileChanged.connect(self.setValue)
        return editor
    
    def setEditorData(self, editor, data):
        '''Sets the data to be displayed and edited by the editor from 
        the data model item.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        data : object
                Data to set in the editor
        
        :Returns:
        
        val : bool
              True if new value was set
        '''
        
        _logger.debug("FilenameProperty type %s"%(data.__class__))
        editor.setCurrentFilename(data)
    
    def testValid(self, data):
        '''
        '''
        
        if self.filetype == 'file-list':
            for f in data.split(','):
                #print f, len(glob.glob(f))
                if len(glob.glob(f)) == 0: return False
        elif self.filetype == 'open':
            if data.find(',') != -1: return False
            if len(glob.glob(data)) == 0: return False
            if glob.glob(data)[0] != data: return False
        else:# self.filetype == 'save':
            if data.find(',') != -1: return False
            if os.path.dirname(data) != "" and len(glob.glob(os.path.dirname(data))) == 0: return False
            if os.path.dirname(data) != "" and glob.glob(os.path.dirname(data))[0] != os.path.dirname(data): return False
        return True
    
    def editorData(self, editor):
        ''' Get the data from a finished editor.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        
        :Returns:
        
        val : object
              Data from the editor
        '''
        
        #return editor.currentIndex()
        return editor.selectedFilename()
    
    @qtSlot('const QString&')
    def setValue(self, value):
        '''Set the value for the property
        
        :Parameters:
        
        value : QObject
               Value to store
        '''
        
        _logger.debug("setValue Qstring")
        if value is not None: 
            if value and not self.testValid(value): return False
            valid = True if value else False
            Property.setValue(self, value, valid)
    
    def value(self, role = QtCore.Qt.UserRole):
        ''' Get the value for the given role
        
        :Parameters:
        
        role : enum
               Stored value role (Unused)
        
        :Returns:
        
        value : object
                Stored value
        '''
        
        return str(Property.value(self, role))



"""
class WorkflowProperty(Property):
    '''Connect a font property to a Workflow dialog
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    hints : dict
            GUI hints
    doc : str
          GUI help string
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, hints={}, doc="", parent=None):
        "Initialize a Choice Property"
        
        Property.__init__(self, name, group, property, hints, doc, parent)
        self.operations = self.hints["operations"]
    
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
        extended : object
                   Additional meta information
        parent : QObject
               Parent object
        
        :Returns:
        
        val : BoolProperty
              Property object
        '''
        
        name, hints,doc = parseHints(name, extended)
        _logger.debug("Create WorkflowProperty: %s - %s - %s"%(name, str(property.property(name).__class__), str(hints)))
        if isinstance(property.property(name), basestring) and 'operations' in hints:
            return cls(name, group, property, hints, doc, parent)
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
        data : object
                Data to set in the editor
        
        :Returns:
        
        val : bool
              True if new value was set
        '''
        
        _logger.debug("FilenameProperty type %s"%(data.__class__))
        if isinstance(data, basestring):
            editor.setWorkflow(str(data).split(','))
            return True
        else:
            return False
    
    def editorData(self, editor):
        ''' Get the data from a finished editor.
        
        :Parameters:
    
        editor : QWidget
                 Editor widget to display data
        
        :Returns:
        
        val : object
              Data from the editor
        '''
        
        return ",".join(editor.workflow())
    
    @qtSlot(list)
    @qtSlot(str) #PyQt_PyObject
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
        
        value : object
                Stored value
        '''
        
        return Property.value(self, role)
"""

class StringProperty(Property):
    '''Connect a String property to a QLineEdit
        
    :Parameters:
    
    name : str
           Name of the property
    group : int
            Group index
    property : QObject
               Property object
    hints : dict
            GUI hints
    doc : str
          GUI help string
    parent : QObject
           Parent object
    '''
    
    __metaclass__ = register_property
    
    def __init__(self, name, group, property=None, hints={}, doc="", parent=None):
        "Initialize a String Property"
        
        Property.__init__(self, name, group, property, hints, doc, parent)
    
    def isValid(self):
        ''' Test if the property holds a valid value
        
        :Returns:
        
        val : bool
              True by default for most properties
        '''
        
        if not self.required: return True
        return self.value() != ""
    
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
        extended : object
                   Additional meta information
        parent : QObject
               Parent object
        
        :Returns:
        
        val : StringProperty
              Property object
        '''
        
        name, hints,doc = parseHints(name, extended)
        _logger.debug("Create StringProperty: %s - %s"%(name, str(property.property(name).__class__)))
        if isinstance(property.property(name), basestring) or isinstance(property.property(name), list):
            return cls(name, group, property, hints, doc, parent)
        return None
    
    def setValue(self, value):
        '''Set the value for the property
        
        :Parameters:
        
        value : QObject
               Value to store
        '''
        
        if value is not None:
            valid = True if value else False
            #if self.required: 
            '''
            if value:
                self.propertyValidity.emit(self, True)
            else:
                self.propertyValidity.emit(self, False)
            '''
            Property.setValue(self, str(value), valid)
    
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

