''' Create a Graphical User Interface (GUI) to edit settings of a program

.. todo:: string list editor
.. todo:: filename list editor
.. todo:: refinement, double list editor link with header
.. todo:: spider document editor, link to header
.. todo:: volume mask separate out choice and filename
.. todo:: html help in tab or popup
.. todo:: maximum length for string argument

.. Created on Sep 3, 2012
.. codeauthor:: robertlanglois
'''
#from ..app import tracing
import logging, sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from PyQt4 import QtGui, QtCore
    from dialogs.PropertyDialog import Dialog
    from property import pyqtProperty
    import arachnid
    QtGui;
except:
    QtGui=None
    #tracing.log_import_error("Failed to import PyQT4 module - certain functionality will not be available - graphical user interface", _logger)

def display(parser, name=None, ui=False, config_file="", **extra):
    ''' Display a settings editor graphical user interface
    
    :Parameters:
        
    parser : OptionParser
             The option parser used to parse the command line parameters
    name : str, optional
           Name for the tab, if unspecified then generate from the groups
    '''
    
    if QtGui is None or not ui: return
    app = QtGui.QApplication([])
    QtCore.QCoreApplication.setOrganizationName("Frank Lab")
    QtCore.QCoreApplication.setOrganizationDomain(arachnid.__url__)
    QtCore.QCoreApplication.setApplicationName(arachnid.__project__)
    QtCore.QCoreApplication.setApplicationVersion(arachnid.__version__)
    # Read given config file, command line - open button
    dialog = Dialog()
    dialog.setWindowTitle(config_file)
    tree = parser.create_property_tree(pyqtProperty.pyqtProperty, QtCore.QObject)
    if name is None:
        for branch in tree:
            dialog.addProperty(branch, branch.DisplayName)  
    else:
        dialog.addProperty(tree, name)
    dialog.show()
    # call hook to write new config file
    # set create_cfg
    # app.exec_()
    # or
    sys.exit(app.exec_())

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    
    if QtGui is None: return 
    parser.add_option("-X",  ui=False, help="Display the graphical user interface", gui=dict(nogui=True))
    # Launcher command option
 
