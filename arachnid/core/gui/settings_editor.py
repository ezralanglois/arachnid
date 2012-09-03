''' Create a Graphical User Interface (GUI) to edit settings of a program

.. Created on Sep 3, 2012
.. codeauthor:: robertlanglois
'''
from ..app import tracing
import logging, sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from PyQt4 import QtGui, QtCore
    from dialogs.PropertyDialog import Dialog
    import arachnid
    QtGui;
except:
    QtGui=None
    tracing.log_import_error("Failed to import PyQT4 module - certain functionality will not be available - graphical user interface", _logger)

def display(parser, values):
    ''' Display a settings editor graphical user interface
    
    :Parameters:
        
    parser : OptionParser
             The option parser used to parse the command line parameters
    values : object
             Value object
    '''
    
    if QtGui is None: return
    app = QtGui.QApplication([])
    QtCore.QCoreApplication.setOrganizationName("Frank Lab")
    QtCore.QCoreApplication.setOrganizationDomain(arachnid.__url__)
    QtCore.QCoreApplication.setApplicationName(arachnid.__project__)
    QtCore.QCoreApplication.setApplicationVersion(arachnid.__version__)
    # Read given config file, command line
    dialog = Dialog()
    dialog.show()
    # call hook to write new config file
    sys.exit(app.exec_())

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    
    if QtGui is None: return 
    parser.add_option("-X",  ui=False, help="Display the graphical user interface", gui=dict(nogui=True))
 
