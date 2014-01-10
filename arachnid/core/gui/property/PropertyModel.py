'''Model for a QTreeView.

.. note::

    Adopted from http://qt-apps.org/content/show.php/QPropertyEditor?content=68684
    Original Author: Volker Wiendl with Enhancements by Roman alias banal

.. Created on Dec 2, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..util.qt4_loader import QtGui, QtCore, qtProperty, qtSignal
from Property import Property
import logging, types

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class PropertyModel(QtCore.QAbstractItemModel):
    ''' Defines the property data maintained by a QTreeView
        
    :Parameters:
    
    parent : QObject
             Parent of the Property Model
    '''
    
    propertyValidity = qtSignal(object, object, bool)
    
    def __init__(self, parent=None):
        "Initialize the Property Model"
        
        QtCore.QAbstractItemModel.__init__(self, parent)
        self.rootItem = Property("Root", parent=self)
        self.userCallbacks = []
        self.background_colors = [QtGui.QColor.fromRgb(250, 191, 143), 
                                  QtGui.QColor.fromRgb(146, 205, 220),
                                  QtGui.QColor.fromRgb(149, 179, 215),
                                  QtGui.QColor.fromRgb(178, 161, 199),
                                  QtGui.QColor.fromRgb(194, 214, 155),
                                  QtGui.QColor.fromRgb(217, 149, 148),
                                  QtGui.QColor.fromRgb(84, 141, 212),
                                  QtGui.QColor.fromRgb(148, 138, 84)]
        
    def addOptionList(self, option_list):
        ''' Add options from list/dict format
        '''
        
        from arachnid.core.app import settings
        parser = settings.OptionParser('', version='0.0.1', description="")
        for option in option_list:
            parser.add_option("", **option)
        values = parser.get_default_values()
        names = vars(values).keys()
        self.addOptions(parser.get_config_options(), parser.option_groups, values)
        return values, names
    
    def addOptions(self, option_list, option_groups, option_values, parent=None, rindex=0):
        ''' Add command line options to the Property Tree Model
        
        :Parameters:
        
        option_list, option_groups, option_values
        '''
        
        if parent is None: parent = self.rootItem
        self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount()+len(option_list)+len(option_groups))
        
        if not hasattr(option_values, 'setProperty'):
            def setProperty(obj, key, val): setattr(obj, key, val)
            option_values.setProperty = types.MethodType( setProperty, option_values )
        if not hasattr(option_values, 'property'):
            def property(obj, key): return getattr(obj, key)
            option_values.property = types.MethodType( property, option_values )
        
        for option in option_list:
            for propertyClass in Property.PROPERTIES:
                p = propertyClass.create(option, rindex, option_values, parent=parent)
                if p is not None: break
            if p is None: print 'Error', option.dest
            assert(p is not None)
            #if p.required: 
            p.propertyValidity.connect(self.firePropertyValidity)
        
        for group in option_groups:
            if group.is_child() and len(group.get_config_options()) > 0:
                current = Property(group.title, rindex, parent=parent)
                self.addOptions(group.get_config_options(), group.option_groups, option_values, current)
        
        # groups
        self.endInsertRows()
    
    def firePropertyValidity(self, prop, valid):
        '''
        '''
        
        self.propertyValidity.emit(self, prop, valid)
    
    def _addItems(self, properties, parentItem, rindex=0):
        '''Recursively add external properties to the model
        
        :Parameters:
        
        properties : list
                     List of properties
        parentItem : Property
                     Parent of current property set
        rindex : int
                Current row index for grouping
        '''
            
        for propertyObject in properties:
            #name = getattr(propertyObject.__class__, 'DisplayName', propertyObject.__class__.__name__) #.replace('_', '-'))
            name = propertyObject.__class__.__name__
            _logger.debug("Add item %s - %d - child: %d"%(name, rindex, len(propertyObject._children)))
            currentItem = Property(name, rindex, None, None, parentItem)
            
            props=[]
            for propName in dir(propertyObject.__class__):
                metaProperty = getattr(propertyObject.__class__, propName)
                if not issubclass(metaProperty.__class__, qtProperty): continue
                props.append(propName)
            
            '''
            props = dir(propertyObject.__class__)
            oprops = list(props)
            last = -1
            for propName in oprops:
                metaProperty = getattr(propertyObject.__class__, propName)
                if not issubclass(metaProperty.__class__, qtProperty): continue
                try:
                    props[metaProperty.order_index]=propName
                    if metaProperty.order_index > last: last = metaProperty.order_index
                except:
                    _logger.error("%s - %d < %d"%(propName, metaProperty.order_index, len(oprops)))
                    raise
            props = props[:last+1]
            '''
            
            props = [prop for prop in props if isinstance(getattr(propertyObject.__class__, prop), qtProperty) ]
            #props = [prop for prop in dir(propertyObject.__class__) if isinstance(getattr(propertyObject.__class__, prop), qtProperty) ]
            
            
            
            for propName in props:
                metaProperty = getattr(propertyObject.__class__, propName)
                for propertyClass in Property.PROPERTIES:
                    p = propertyClass.create(propName, rindex, propertyObject, metaProperty, currentItem)
                    if p is not None: break
                if p is not None:
                    p.propertyValidity.connect(self.firePropertyValidity)
                row = p.row() if p is not None else -1
                _logger.debug("Add property %s - %d"%(propName,row))
            self._addItems(propertyObject._children, currentItem, rindex)
            if parentItem == self.rootItem: rindex += 1
    
    def addItem(self, propertyObject):
        ''' Add a property object to the tree
        
        :Parameters:
        
        propertyObject : QObject
                        Property object to monitor
        '''
        
        _logger.debug("Add item: %s"%str(propertyObject.__class__.__name__))
        if hasattr(propertyObject, 'todict') or hasattr(propertyObject, '_children'):
            #if hasattr(propertyObject, '_children'): propertyObject = [propertyObject]
            
            props = [prop for prop in dir(propertyObject.__class__) if isinstance(getattr(propertyObject.__class__, prop), qtProperty)]
            if len(props) != 0:
                _logger.debug("Tab as root")
                if hasattr(propertyObject, '_children'): propertyObject = [propertyObject]
            else:
                if hasattr(propertyObject, '_children'): propertyObject = propertyObject._children
            self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount()+len(propertyObject))
            self._addItems(propertyObject, self.rootItem)
            self.endInsertRows()
        elif hasattr(propertyObject, 'external_properties'):
            # Insert properties for classes
            self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount()+len(propertyObject.external_properties))
            self._addItems(propertyObject.external_properties, self.rootItem)
            self.endInsertRows()
        else:
            # Create property/class hierarchy
            propertyMap = {}
            classList = []
            pypropertyMap = dict([(prop, getattr(propertyObject.__class__, prop)) for prop in dir(propertyObject.__class__) if isinstance(getattr(propertyObject.__class__, prop), qtProperty) ])
            classMap = {}
            classObject = propertyObject.__class__
            metaObject = propertyObject.metaObject()
            while metaObject is not None:
                for i in xrange(metaObject.propertyOffset(), metaObject.propertyCount()):
                    prop = metaObject.property(i)
                    if not prop.isUser(): continue
                    propertyMap[prop] = metaObject
                classList.append(metaObject)
                classMap[metaObject.className()] = classObject
                metaObject = metaObject.superClass()
                classObject = classObject.__bases__[0]
            
            # Remove empty classes
            finalClassList = []
            for obj in classList:
                found = False
                for obj2 in propertyMap.itervalues():
                    if obj2 == obj:
                        found = True
                        break
                if found: finalClassList.append(obj)
            
            # Insert properties for classes
            self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount()+len(finalClassList))
            
            propertyItem = None
            for rindex, metaObject in enumerate(finalClassList):
                name = metaObject.className()
                name = getattr(classMap[name], 'DisplayName', name)
                propertyItem = Property(name, rindex, None, None, self.rootItem)
                keys = propertyMap.keys()
                keys.sort()
                #for prop, obj in propertyMap.iteritems():
                for prop in keys:
                    obj = propertyMap[prop]
                    if obj != metaObject: continue
                    metaProperty = QtCore.QMetaProperty(prop)
                    try:
                        extProperty = pypropertyMap[prop.name()]
                    except:
                        print 'Error', prop.name(), "-->", pypropertyMap
                        raise
                    p = None
                    for propertyClass in Property.PROPERTIES:
                        p = propertyClass.create(metaProperty.name(), rindex, propertyObject, extProperty, propertyItem)
                        if p is not None: break
                    if p is not None:
                        p.propertyValidity.connect(self.firePropertyValidity)
            self.endInsertRows()
            if propertyItem: self.addDynamicProperties(propertyItem, propertyObject)
    
    def updateItem(self, propertyObject, parent = QtCore.QModelIndex()):
        '''Update a property object in the tree
        
        :Parameters:
        
        propertyObject : QObject
                        Property object to monitor
        parent : QModelIndex
                 Index of parent in the tree
        '''
        
        parentItem = self.rootItem
        if parent.isValid():
            parentItem = parent.internalPointer()
        if parentItem.property_object() != propertyObject:
            parentItem = parentItem.findPropertyObject(propertyObject)
        if parentItem:
            itemIndex = self.createIndex(parentItem.row(), 0, parentItem)
            self.dataChanged(itemIndex, self.createIndex(parentItem.row(), 1, parentItem))
            dynamicProperties = propertyObject.dynamicPropertyNames()
            children = parentItem.parent().children()
            removed = 0
            for i in xrange(len(children)):
                obj = children[i]
                if obj.property("__Dynamic").toBool() or dynamicProperties.contains(obj.objectName()): continue
                self.beginRemoveRows(itemIndex.parent(), i - removed, i - removed)
                removed += 1
                self.endRemoveRows()
            self.addDynamicProperties(parentItem.parent(), propertyObject)
    
    def addDynamicProperties(self, parent, propertyObject):
        ''' Add dynamic properties to the tree
        
        :Parameters:
        
        parent : Property
                 Parent of the Property object to monitor
        propertyObject : QObject
                         Property object to maintain
        '''
        
        dynamicProperties = propertyObject.dynamicPropertyNames()
        for child in parent.children():
            if not child.property("__Dynamic").toBool() : continue
            index = dynamicProperties.indexOf( child.objectName() )
            if index != -1:
                dynamicProperties.removeAt(index)
                continue
        i = 0
        while i < len(dynamicProperties):
            if dynamicProperties[i].startsWith("_") or propertyObject.property( dynamicProperties[i] ).isValid(): dynamicProperties.removeAt(i)
            else: i+= 1
        
        if len(dynamicProperties) == 0: return
        
        parentIndex = self.createIndex(parent.row(), 0, parent)
        rows = self.rowCount(parentIndex)
        self.beginInsertRows(parentIndex, rows, rows + dynamicProperties.count() - 1 )
        
        for dynProp in dynamicProperties:
            #v = propertyObject.property(dynProp)
            p = None
            if len(self.userCallbacks) > 0:
                for callback in self.userCallbacks:
                    p = callback(property.name(), propertyObject, parent)
                    if p is not None: break
            if p is None: p = Property(dynProp, propertyObject, parent)
            p.setProperty("__Dynamic", True)
        self.endInsertRows()
    
    def clear(self):
        ''' Clear the property tree and remove from model
        '''
        
        self.beginRemoveRows(QtCore.QModelIndex(), 0, self.rowCount())
        self.rootItem = Property("Root", None, self)
        self.endRemoveRows()
    
    #### overridden virtual functions
    
    def buddy(self, index):
        ''' Returns a model index for the buddy of the item represented by index. When the user wants 
        to edit an item, the view will call this function to check whether another item in the model 
        should be edited instead. Then, the view will construct a delegate using the model index returned 
        by the buddy item.
        
        :Parameters:
        
        index : QModelIndex
                Index of a model item
        
        :Returns:
        
        val : QModelIndex
              Index of the buddy
        '''
        
        if index.isValid() and index.column() == 0:
            return self.createIndex(index.row(), 1, index.internalPointer())
        return index
    
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
            if section == 0: return self.tr("Name")
            if section == 1: return self.tr("Value")
        return None
    
    def flags(self, index):
        '''Returns the item flags for the given index
        
        :Parameters:
        
        index : QModelIndex
                Index of a model item
        
        :Returns:
        
        val : ItemFlags
              Flags for item at index
        '''
        
        if not index.isValid(): return QtCore.Qt.ItemIsEnabled
        item = index.internalPointer()
        
        if item.isRoot():       flags = QtCore.Qt.ItemIsEnabled
        elif item.isReadOnly(): flags = QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsSelectable
        else:                   flags = QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable
        if item.isBool():       flags = flags | QtCore.Qt.ItemIsUserCheckable
        return flags
    
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        ''' Sets the role data for the item at index to value
        
        :Parameters:
        
        index : QModelIndex
                Index of a model item
        value : object
                New value for item at index with role
        role : enum
               Data role
        
        :Returns:
        
        val : bool
              True if successful
        '''
        
        if index.isValid():
            if role == QtCore.Qt.EditRole:
                item = index.internalPointer()
                item.setValue(value)
                self.emit(QtCore.SIGNAL("dataChanged (const QModelIndex&,const QModelIndex&)"), index, index)
                return True
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
        item = index.internalPointer()
        
        if role == QtCore.Qt.ForegroundRole and index.column() == 0 and 'required' in item.hints and item.hints['required']:
            return QtGui.QColor(QtCore.Qt.blue)
        if role == QtCore.Qt.BackgroundRole and index.column() == 1 and 'required' in item.hints and item.hints['required']:
            if not item.isValid():
                return QtGui.QColor('#F0E68C')
        
        if (role == QtCore.Qt.ToolTipRole or role == QtCore.Qt.StatusTipRole) and item.doc is not None:
            return "<FONT COLOR=black>"+item.doc+"<FONT>"
        elif role == QtCore.Qt.ToolTipRole or role == QtCore.Qt.DecorationRole or role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            if index.column() == 0: return item.displayName
            if index.column() == 1: return item.value(role)
        #if role == QtCore.Qt.BackgroundRole:
        #    if item.isRoot(): return QtGui.QApplication.palette("QTreeView").brush(QtGui.QPalette.Normal, QtGui.QPalette.Button).color()
            '''
                group = item.group
                while group > len(self.background_colors): group -= len(self.background_colors)
                color = self.background_colors[group]
                return QtGui.QBrush(color)
            '''
        if role == QtCore.Qt.CheckStateRole:
            if index.column() == 1 and item.isBool(): return item.value(role)
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
        
        parentItem = self.rootItem
        if parent.isValid(): parentItem = parent.internalPointer()
        return len(parentItem.children())
    
    def maximumTextWidth(self, fontMetric, indent, column=0, parent = QtCore.QModelIndex()):
        ''' Get the maximum text with of a table column
        
        :Parameters:
        
        fontMetric : QFontMetrics
                     Font metric used to measure text width
        indent : int
                 Width of indent
        column : int
                 Column index
        parent : QModelIndex
                 Parent of item
        
        :Returns:
        
        val : int
              Maximum width of text in a column
        '''
        
        parentItem = self.rootItem
        if parent.isValid(): parentItem = parent.internalPointer()
        return parentItem.maximumNameWidth(fontMetric, indent)
    
    def columnCount(self, parent = QtCore.QModelIndex()):
        ''' Get the number of columns - 2
        
        :Parameters:
    
        parent : QModelIndex
                 Parent of item
        
        :Returns:
        
        val : int
              Return value: 2
        '''
        
        return 2
    
    def parent(self, index):
        '''Returns the parent of the model item with the given index. If the item has no parent, an invalid QModelIndex is returned
        
        :Parameters:
        
        index : QModelIndex
                Index of a model item
        
        :Returns:
        
        index : QModelIndex
                Parent of the model item
        '''
        
        if not index.isValid(): return QtCore.QModelIndex()
        childItem = index.internalPointer()
        parentItem = childItem.parent()
        if parentItem is None or parentItem == self.rootItem: return QtCore.QModelIndex()
        return self.createIndex(parentItem.row(), 0, parentItem)
    
    def index(self, row, column, parent = QtCore.QModelIndex()):
        ''' Returns the index of the item in the model specified by the given row, column and parent index
        
        :Parameters:
        
        row : int
              Row of item
        column : int
                Column of item
        parent : QModelIndex
                 Parent of item
        
        :Returns:
        
        index : QModelIndex
                Current index of item
        '''
        
        parentItem = self.rootItem
        if parent.isValid(): 
            parentItem = parent.internalPointer()
        if row >= len(parentItem.children()) or row < 0: 
            return QtCore.QModelIndex()
        return self.createIndex(row, column, parentItem.children()[row])
    
    def saveState(self, settings, parent = QtCore.QModelIndex()):
        ''' Save the state of the properties in the tree
        
        :Parameters:
        
        settings : QSettings
                   Save settings to platform specific location
        parent : QModelIndex
                 Parent of item
        '''
        
        parentItem = self.rootItem
        if parent.isValid(): parentItem = parent.internalPointer()
        parentItem.saveState(settings)
    
    def restoreState(self, settings, parent = QtCore.QModelIndex()):
        ''' Restore the state of the properties in the tree
        
        :Parameters:
        
        settings : QSettings
                   Load settings from platform specific location
        parent : QModelIndex
                 Parent of item
        '''
        
        parentItem = self.rootItem
        if parent.isValid(): parentItem = parent.internalPointer()
        parentItem.restoreState(settings)
    
    def totalInvalid(self):
        ''' Count the total number of children under this node.
        
        :Returns:
        
        total : int
                Total number of invalid, but required children
        '''
        
        return self.rootItem.totalInvalid()
    

