''' Project Creation Wizard

.. Created on Dec 8, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
import logging, sys
import arachnid

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from util.qt4_loader import QtGui, QtCore
    QtGui;
except:
    QtGui=None
    tracing.log_import_error("Failed to import PyQT4 module - certain functionality will not be available - graphical user interface", _logger)
    _logger.exception("message")
else:
    from dialogs.ProjectWizard import MainWindow as Wizard

def main():
    # Create GUI and display

    tracing.configure_logging()
    if QtGui is None:
        _logger.error("PyQT4 not installed")
        tracing.print_import_warnings()
        sys.exit(1)
    app = QtGui.QApplication([])
    QtCore.QCoreApplication.setOrganizationName("Frank Lab")
    QtCore.QCoreApplication.setOrganizationDomain(arachnid.__url__)
    QtCore.QCoreApplication.setApplicationName(arachnid.__project__+'.project_wizard')
    QtCore.QCoreApplication.setApplicationVersion(arachnid.__version__)
    dialog = Wizard() 
    dialog.show()
    sys.exit(app.exec_())
