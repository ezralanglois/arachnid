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
#from util import messagebox
import logging
import sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class Dialog(QtGui.QDialog): 
    ''' Automated GUI build from command line options
    '''
    
    def __init__(self, program, parent=None):
        "Initialize screener window"
        
        QtGui.QDialog.__init__(self, parent)
        
        # Build window
        _logger.info("Building main window ...")
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.settingsTabWidget = TabWidget(self)
        self.ui.configTabLayout = QtGui.QHBoxLayout(self.ui.configTab)
        self.ui.configTabLayout.addWidget(self.ui.settingsTabWidget)
        self.ui.settingsTabWidget.controlValidity.connect(self.setValid)
        
        self.ui.monitorWidget = MonitorWidget(self)
        self.ui.runTabLayout = QtGui.QHBoxLayout(self.ui.runTab)
        self.ui.runTabLayout.addWidget(self.ui.monitorWidget)
        #self.ui.monitorWidget.runProgram.connect(self.runProgram)
        #self.ui.monitorWidget.monitorProgram.connect(self.monitorProgram)
        self.program=program
        
        self.ui.settingsTabWidget.addSettings(*program.settings())
        self.ui.monitorWidget.setWorkflow(program)
        self.ui.monitorWidget.setLogFile(program.values.log_file)
    
    def setValid(self, widget, valid):
        '''
        '''
        
        try:
            self.program.check_options_validity()
        except:
            _logger.exception("Error while checking options")
            #messagebox.exception_message(self, "Error in options")
            self.ui.tabWidget.setCurrentIndex(0)
            self.ui.tabWidget.setTabEnabled(1, False)
        else:
            self.ui.tabWidget.setTabEnabled(1, valid)

def display(program, display_gui=False, **extra):
    '''
    '''
    
    if not display_gui: return
    logging.getLogger().setLevel(logging.INFO)
    from util import qtapp
    app = qtapp.create_app()
    if app is None:
        _logger.error("PyQT4 not installed")
        sys.exit(1)
    program.ensure_log_file()
    dialog = Dialog(program)
    dialog.setWindowTitle(program.name())
    dialog.show()
    sys.exit(app.exec_())

def setup_options(parser, pgroup=None, main_option=False):
    '''
    '''
    if QtGui is None: return 
    from ..app.settings import OptionGroup
    if pgroup is None: pgroup=parser
    group = OptionGroup(parser, "User Interface", "Options to control the state of the AutoGUI", id=__name__)
    group.add_option("-X", display_gui=False,       help="Display the graphical user interface", gui=dict(nogui=True), dependent=False)
    pgroup.add_option_group(group)

