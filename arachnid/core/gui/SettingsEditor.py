''' A settings editor tab widget

.. Created on Nov 1, 2013
.. codeauthor:: robertlanglois
'''
from util.qt4_loader import QtGui, qtSignal
from pyui.SettingsEditor import Ui_TabWidget
import property 
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class TabWidget(QtGui.QTabWidget):
    '''
    '''
    
    controlValidity = qtSignal(object, bool)
    
    def __init__(self, parent=None):
        '''
        '''
        
        QtGui.QTabWidget.__init__(self, parent)
        self.ui = Ui_TabWidget()
        self.ui.setupUi(self)
        property.setView(self.ui.propertyTreeView)
        self.tree_views={}
        self.model_tab={}
        self.invalid_count=0
        # Taken from http://stackoverflow.com/questions/12438095/qt-vertical-scroll-bar-stylesheets
        self.treeViewStyle='''QScrollBar:vertical { 
        border: 1px solid #999999;
        background:white;
        width:10px;
        margin: 0px 0px 0px 0px;
        }
        QScrollBar::handle:vertical {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,stop: 0  rgb(32, 47, 130), stop: 0.5 rgb(32, 47, 130),  stop:1 rgb(32, 47, 130));
        min-height: 0px;
        }
        QScrollBar::add-line:vertical {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,stop: 0  rgb(32, 47, 130), stop: 0.5 rgb(32, 47, 130),  stop:1 rgb(32, 47, 130));
        height: 0px;
        subcontrol-position: bottom;
        subcontrol-origin: margin;
        }
        QScrollBar::sub-line:vertical {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,stop: 0  rgb(32, 47, 130), stop: 0.5 rgb(32, 47, 130),  stop:1 rgb(32, 47, 130));
        height: 0px;
        subcontrol-position: top;
        subcontrol-origin: margin;
        }
        QTreeView::item {
            border: 1px solid #d9d9d9;
        }
        QTreeView::item:selected {
            border-color:green; 
            border-style:outset; 
            border-width:2px; 
            color:black; 
        }
        '''
    
    def addSettings(self, option_list, group_list, values):
        '''
        '''
        
        for group in group_list:
            if group.is_child(): continue
            treeView = self.copyTreeView()
            property.setView(treeView)
            self.tree_views[group.title]=treeView
            treeView.model().addOptions(group.get_config_options(), group.option_groups, values)
            icon = QtGui.QIcon()
            #print group.title, treeView.model().totalInvalid()
            if treeView.model().totalInvalid() > 0:
                icon.addPixmap(QtGui.QPixmap(":/mini/mini/exclamation.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.invalid_count += 1
            else:
                icon.addPixmap(QtGui.QPixmap(":/mini/mini/accept.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.model_tab[treeView.model()]=self.count()-1
            self.addTab(treeView, icon, group.title)
            if treeView.model().totalInvalid() > 0:
                treeView.model().propertyValidity.connect(self.updateValid)
            width = treeView.model().maximumTextWidth(self.fontMetrics(), treeView.indentation())
            treeView.setColumnWidth(0, width)
        self.removeTab(0)
        #self.ui.propertyTreeView.model().addOptions(option_list, [], values)
        if not self.isValid(): self.controlValidity.emit(self, False)
    
    def isValid(self):
        '''
        '''
        
        return self.invalid_count == 0
    
    def updateValid(self, model, prop, valid):
        '''
        '''
        
        if valid:
            if model.totalInvalid() == 0:
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap(":/mini/mini/accept.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.setTabIcon(self.model_tab[model], icon)
                self.invalid_count -= 1
                if self.isValid(): self.controlValidity.emit(self, True)
        else:
            if model.totalInvalid() == 1:
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap(":/mini/mini/exclamation.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.setTabIcon(self.model_tab[model], icon)
                self.invalid_count += 1
                if not self.isValid(): self.controlValidity.emit(self, False)
    
    def showEvent(self, event_obj):
        '''Called before the dialog is shown
        
        :Parameters:
        
        event_obj : QShowEvent
                Event object
        '''
        
        for view in self.tree_views.values(): 
            view.model().reset()
            view.expandAll()
        QtGui.QTabWidget.showEvent(self, event_obj)
    
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
        treeView.setEditTriggers(self.ui.propertyTreeView.editTriggers())
        treeView.setStyleSheet(self.treeViewStyle)
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

    