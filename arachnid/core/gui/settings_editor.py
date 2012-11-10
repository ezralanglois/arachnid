''' Create a Graphical User Interface (GUI) to edit settings of a program

.. todo:: string list editor
.. todo:: filename list editor
.. todo:: refinement, double list editor link with header
.. todo:: spider document editor, link to header
.. todo:: volume mask separate out choice and filename
.. todo:: html help in tab or popup
.. todo:: add logging debug more
.. todo:: set style sheet and mark required items with background color

.. Created on Sep 3, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
import logging, sys, os

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
    tracing.log_import_error("Failed to import PyQT4 module - certain functionality will not be available - graphical user interface", _logger)

def _create_settings_dialog(parser, options, name=None, config_file="", style_sheet="", **extra):
    ''' Create a setting dialog editor
    
    :Parameters:
    
    parser : OptionParser
             The option parser used to parse the command line parameters
    options : object
              Options container
    name : str, optional
           Name for the tab, if unspecified then generate from the groups
    config_file : str
                  Name of configuration file to display in title
    style_sheet : str
                  Style sheet for the settings dialog
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    app : QApplication
          Application handle
    dialog : QDialog
             Settings editor dialog
    '''
    
    app = QtGui.QApplication([])
    if style_sheet != "" and os.path.exists(style_sheet):
        fin = open(style_sheet, 'r')
        _logger.info("Using style sheet: %s"%style_sheet)
        try:
            sheet = "".join(fin.readlines())
            app.setStyleSheet(sheet)
            print sheet
        finally: fin.close()
    QtCore.QCoreApplication.setOrganizationName("Frank Lab")
    QtCore.QCoreApplication.setOrganizationDomain(arachnid.__url__)
    QtCore.QCoreApplication.setApplicationName(arachnid.__project__)
    QtCore.QCoreApplication.setApplicationVersion(arachnid.__version__)
    # Read given config file, command line - open button
    dialog = Dialog()
    dialog.setWindowTitle(config_file)
    tree = parser.create_property_tree(options, pyqtProperty.PyqtProperty, QtCore.QObject)
    if name is None:
        for branch in tree:
            dialog.addProperty(branch, branch.DisplayName)
    else:
        dialog.addProperty(tree, name)
    return app, dialog

def screenshot(parser, name=None, screen_shot="", **extra):
    ''' Takes a screen shot of the settings dialog editor and writes it
    to a PNG image file.
    
    :Parameters:
    
    parser : OptionParser
             The option parser used to parse the command line parameters
    name : str, optional
           Name for the tab, if unspecified then generate from the groups
    screen_shot : str
                  Output filename for the screen shot
    extra : dict
            Unused keyword arguments
    '''
    
    if QtGui is None or screen_shot == "": return
    app, dialog = _create_settings_dialog(parser, name, **extra)
    dialog.showSetup()
    #originalPixmap = QtGui.QPixmap.grabWindow(QtGui.QApplication.desktop().winId())
    #dialog.show()
    #originalPixmap = QtGui.QPixmap.grabWindow(dialog.winId())
    #dialog.hide()
    originalPixmap = QtGui.QPixmap.grabWidget(dialog)
    originalPixmap.save(os.path.splitext(screen_shot)[0]+'.png', 'png')
    sys.exit(0)

def display(parser, options, name=None, ui=False, **extra):
    ''' Display a settings editor graphical user interface
    
    :Parameters:
        
    parser : OptionParser
             The option parser used to parse the command line parameters
    options : object
              Options container
    name : str, optional
           Name for the tab, if unspecified then generate from the groups
    ui : bool
         Set true to display the dialog
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    create_cfg : str
                 Output filename for the configuration file
    '''
    
    screenshot(parser, name, **extra)
    if QtGui is None or not ui: return extra['create_cfg']
    app, dialog = _create_settings_dialog(parser, options, name, **extra)
    dialog.show()
    ret = app.exec_()
    if dialog.windowTitle() == "": sys.exit(ret)
    return str(dialog.windowTitle())
    

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    
    if QtGui is None: return 
    
    parser.add_option("-C", style_sheet="", help="Input filename for the style sheet the graphical user interface", gui=dict(nogui=True, filetype='open'), dependent=False)
    parser.add_option("-X", ui=False,       help="Display the graphical user interface", gui=dict(nogui=True), dependent=False)
    parser.add_option("-S", screen_shot="", help="Output filename for a screenshot of the UI", gui=dict(filetype="save", nogui=True), dependent=False)
    # Launcher command option
 
