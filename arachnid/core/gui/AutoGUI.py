'''

- Settings
- Monitor
- Image viewer

.. Created on Sep 5, 2013
.. codeauthor:: robertlanglois
'''
from pyui.AutoGUI import Ui_Dialog
from SettingsEditor import TabWidget
from Monitor import Widget as MonitorWidget
from util.qt4_loader import QtGui #, QtCore, qtSlot,QtWebKit
import logging
import sys, os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class Dialog(QtGui.QDialog): 
    ''' Main window display for the plotting tool
    '''
    
    def __init__(self, parent=None):
        "Initialize screener window"
        
        QtGui.QDialog.__init__(self, parent)
        
        root = logging.getLogger()
        while len(root.handlers) > 0: root.removeHandler(root.handlers[0])
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(message)s'))
        h.setLevel(logging.INFO)
        root.addHandler(h)
        
        # Build window
        _logger.info("\rBuilding main window ...")
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.settingsTabWidget = TabWidget()
        self.ui.configTabLayout = QtGui.QHBoxLayout(self.ui.configTab)
        self.ui.configTabLayout.addWidget(self.ui.settingsTabWidget)
        self.ui.settingsTabWidget.controlValidity.connect(self.setValid)
        
        self.ui.monitorWidget = MonitorWidget()
        self.ui.runTabLayout = QtGui.QHBoxLayout(self.ui.runTab)
        self.ui.runTabLayout.addWidget(self.ui.monitorWidget)
        self.ui.monitorWidget.runProgram.connect(self.runProgram)
        self.ui.monitorWidget.monitorProgram.connect(self.monitorProgram)
        
        self.values = None
        self.qout = None
        
    
    def addSettings(self, option_list, group_list, values):
        '''
        '''
        
        self.ui.settingsTabWidget.addSettings(option_list, group_list, values)
        self.values = values
    
    def setQueue(self, qout):
        '''
        '''
        
        self.qout = qout
    
    def setLogFile(self, filename):
        '''
        '''
        
        self.ui.monitorWidget.setLogFile(filename)
    
    def setValid(self, widget, valid):
        '''
        '''
    
        self.ui.tabWidget.setTabEnabled(1, valid)
    
    def monitorProgram(self):
        '''
        '''
        
        if self.qout is not None:
            self.qout.put(None)
            self.qout=None
    
    def runProgram(self):
        '''
        '''
        
        if self.qout is not None:
            #self.values.input_files = list(self.values.input_files)
            values = Values()
            for key, val in vars(self.values).iteritems():
                if callable(val): continue
                if key == 'input_files': 
                    setattr(values, key, list(val))
                else:
                    setattr(values, key, val)
            import pickle
            try:
                pickle.dumps(values)
            except:
                for key, val in vars(values).iteritems():
                    try:
                        pickle.dumps(val)
                    except:
                        print 'Failed: ', key, val
                self.qout.put(None)
            else:
                self.qout.put(values)
            self.qout=None

class Values:
    pass

def display_mp(name, parser, values, display_gui=False, **extra):
    '''
    '''
    
    if not display_gui: return values
    def display_worker(name, parser, values, display_gui, qout, extra):
        return display(name, parser, values, display_gui, qout, **extra)
    
    import multiprocessing
    qout = multiprocessing.Queue()
    displayProcess = multiprocessing.Process(target=display_worker, args=(name, parser, values, display_gui, qout, extra))
    displayProcess.start()
    
    newvalues = qout.get()
    newvalues.input_files = values.input_files.make(newvalues.input_files)
    return newvalues

def display(name, parser, values, display_gui=False, qout=None, **extra):
    '''
    '''
    
    if not display_gui: return values
    logging.getLogger().setLevel(logging.INFO)
    from util import qtapp
    app = qtapp.create_app()
    if app is None:
        _logger.error("PyQT4 not installed")
        sys.exit(1)
    dialog = Dialog()
    dialog.setWindowTitle(name)
    if qout is not None: dialog.setQueue(qout)
    if values.log_file == "":
        values.log_file = os.path.basename(sys.argv[0])+".log"
    dialog.setLogFile(values.log_file)
    dialog.addSettings(parser.get_config_options(), parser.option_groups, values)
    dialog.show()
    app.exec_()
    if qout is not None: qout.put(None)
    return values

def setup_options(parser, pgroup=None, main_option=False):
    '''
    '''
    if QtGui is None: return 
    from ..app.settings import OptionGroup
    if pgroup is None: pgroup=parser
    group = OptionGroup(parser, "User Interface", "Options to control the state of the AutoGUI", id=__name__)
    group.add_option("-X", display_gui=False,       help="Display the graphical user interface", gui=dict(nogui=True), dependent=False)
    pgroup.add_option_group(group)

