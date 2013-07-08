''' Displays a set of tabbed property trees

.. Created on Dec 3, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..util.qt4_loader import QtCore, QtGui
try: from pyui.PropertyDialog import Ui_PropertyDialog
except:
    raise
    #from PyQt4 import uic
    #Ui_PropertyDialog = os.path.join(os.path.split(__file__)[0], "PropertyDialog.ui")
from .. import property
import logging, os

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
        if isinstance(Ui_PropertyDialog, str):
            self.ui = None
            #self.ui = uic.loadUi(Ui_PropertyDialog, self)
        else:
            self.ui = Ui_PropertyDialog()
            self.ui.setupUi(self)
        
        self.ui.tabWidget.removeTab(0)
        property.setView(self.ui.propertyTreeView)
        self.treeViews = []
        
    def showSetup(self):
        ''' Setup the dialog before showing it
        '''
        
        for view in self.treeViews: view.expandAll()
    
    def showEvent(self, event_obj):
        '''Called before the dialog is shown
        
        :Parameters:
        
        event_obj : QShowEvent
                Event object
        '''
        
        self.showSetup()
        QtGui.QDialog.showEvent(self, event_obj)
    
    def done(self, mode):
        '''Dialog is done
        
        :Parameters:
        
        mode : int
               Dialog accept mode
        '''
        
        filename = self.windowTitle()
        if os.path.exists(filename):
            ret = QtGui.QMessageBox.warning(self, "Warning", "Overwrite existing file?", QtGui.QMessageBox.Yes| QtGui.QMessageBox.No)
            if ret == QtGui.QMessageBox.No: filename = ""
        if filename == "":
            filename = QtGui.QFileDialog.getSaveFileName(self, self.tr("Save config file as"), QtCore.QDir.currentPath(), "Config files (*.cfg)")
            if isinstance(filename, tuple): filename=filename[0]
            if filename: 
                self.setWindowTitle(filename)
            else:
                ret = QtGui.QMessageBox.warning(self, "Warning", "Close without saving?", QtGui.QMessageBox.Yes| QtGui.QMessageBox.No)
                if ret == QtGui.QMessageBox.No: return
        super(Dialog, self).done(mode)
    
    def addProperty(self, obj, name, icon=None):
        ''' Add a property to the property tree
        
        :Parameters:
    
        obj : QObject
              Class to transverse for properties
        name : str
               Name of tab
        icon : QIcon
               Icon on a tab
        '''
        
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
        
        self.treeViews.append(treeView)
        tab.setToolTip(name)
        if isinstance(icon, str): icon = QtGui.QIcon(icon)
        if icon is not None: self.ui.tabWidget.addTab(tab, icon, name)
        else: self.ui.tabWidget.addTab(tab, name)
        
        treeView.model().addItem(obj)
        width = treeView.model().maximumTextWidth(self.fontMetrics(), treeView.indentation())
        treeView.setColumnWidth(0, width)
    
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
                print '**', name, self.ui.propertyTreeView.property(name)
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
        
        for view in self.treeViews: view.model().saveState(settings)
        #self.ui.propertyTreeView.model().saveState(settings)
    
    def restoreState(self, settings):
        ''' Restore the state of the properties in the property tree
        
        :Parameters:
        
        settings : QSettings
                   Load settings from platform specific location
        '''
        
        for view in self.treeViews: view.model().restoreState(settings)
        #self.ui.propertyTreeView.model().restoreState(settings)


