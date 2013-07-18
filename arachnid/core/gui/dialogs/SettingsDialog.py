''' Displays a set of tabbed property trees

.. Created on Jul 16, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..util.qt4_loader import QtGui
from pyui.PropertyDialog import Ui_PropertyDialog
from .. import property
from arachnid.core.app import settings
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class Dialog(QtGui.QDialog):
    ''' Display a properties of an application
    
    :Parameters:
    
    parent : QObject
             Parent object of the dialog
    '''
    
    def __init__(self, parent=None):
        "Initialize the dialog"
        
        QtGui.QDialog.__init__(self, parent)
        self.ui = Ui_PropertyDialog()
        self.ui.setupUi(self)
        self.ui.tabWidget.removeTab(0)
        property.setView(self.ui.propertyTreeView)
        self.treeViews = dict()
        self.options = dict()
        
    def setupDialog(self):
        ''' Setup the dialog before showing it
        '''
        
        for view in self.treeViews.values(): 
            view.model().reset()
            view.expandAll()
    
    def showEvent(self, event_obj):
        '''Called before the dialog is shown
        
        :Parameters:
        
        event_obj : QShowEvent
                Event object
        '''
        
        self.setupDialog()
        QtGui.QDialog.showEvent(self, event_obj)
    
    def addOptions(self, name, option_list, icon=None):
        ''' Add a set of options and groups as a tree
        
        :Parameters:
        
        name : str
               Name of the tab
        option_list : list
                       List of options
        icon : QIcon
               Icon for the tab
        '''
        
        if len(option_list) == 0: return
        
        if name not in self.options:
            self.options[name]=[]
        self.options[name].extend(option_list)
        self.invald=True
        
        if name not in self.treeViews:
            if self.ui.tabWidget.count() > 0:
                _logger.debug("Add new property: %s - %d"%(name, self.ui.tabWidget.count()))
                treeView = self.copyTreeView()
                property.setView(treeView)
                tab = QtGui.QWidget(self)
                horizontalLayout = QtGui.QHBoxLayout(tab)
                horizontalLayout.setContentsMargins(0, 0, 0, 0)
                horizontalLayout.addWidget(treeView)
            else: 
                _logger.debug("Add first property: %s - %d"%(name, self.ui.tabWidget.count()))
                treeView = self.ui.propertyTreeView
                tab = self.ui.tab
            self.treeViews[name] = treeView
            tab.setToolTip(name)
            if isinstance(icon, str): icon = QtGui.QIcon(icon)
            if icon is not None: self.ui.tabWidget.addTab(tab, icon, name)
            else: self.ui.tabWidget.addTab(tab, name)
        else:
            treeView.model().clear()
        
        parser = settings.OptionParser('', version='0.0.1', description=name)
        for option in option_list:
            parser.add_option("", **option)
        values = parser.get_default_values()
        names = vars(values).keys()
        treeView.model().addOptions(parser.get_config_options(), parser.option_groups, values)
        width = treeView.model().maximumTextWidth(self.fontMetrics(), treeView.indentation())
        treeView.setColumnWidth(0, width)
        self.values = values
        return values, names
    
    def copyTreeView(self):
        '''Copy the properties from the original tree view
        
        :Returns:
        
        treeView : QTreeView
                   Copy of current tree view
        '''
        
        treeView = QtGui.QTreeView(self)
        metaObjectFrom = self.ui.propertyTreeView.metaObject()
        metaObjectTo = treeView.metaObject()
        while metaObjectFrom is not None:
            _logger.debug("Class: %s"%metaObjectFrom.className())
            if metaObjectFrom.className() == 'QWidget': break
            for i in xrange(metaObjectFrom.propertyOffset(), metaObjectFrom.propertyCount()):
                name = metaObjectFrom.property(i).name()
                if name == 'editTriggers': continue
                _logger.debug("Copy tree view: "+str(name))
                treeView.setProperty(name, self.ui.propertyTreeView.property(name))
            metaObjectFrom = metaObjectFrom.superClass()
            metaObjectTo = metaObjectTo.superClass()
        return treeView
    
    def saveState(self, settings):
        ''' Save the state of the properties in the property tree
        
        :Parameters:
        
        settings : QSettings
                   Save settings to platform specific location
        '''
        
        for view in self.treeViews.values(): view.model().saveState(settings)
        #self.ui.propertyTreeView.model().saveState(settings)
    
    def restoreState(self, settings):
        ''' Restore the state of the properties in the property tree
        
        :Parameters:
        
        settings : QSettings
                   Load settings from platform specific location
        '''
        
        for view in self.treeViews.values(): view.model().restoreState(settings)
        #self.ui.propertyTreeView.model().restoreState(settings)


