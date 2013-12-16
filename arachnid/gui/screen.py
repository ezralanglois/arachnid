''' Graphical user interface for screening images

.. Created on Jul 17, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.gui.util import qtapp
from ..core.gui.util.qt4_loader import QtGui
from ..core.app import tracing
import logging, sys, glob, os
from ..core.app import settings
from ..core.metadata import spider_utility

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def main():
    # Create GUI and display
    
    tracing.configure_logging()
    app = qtapp.create_app()
    if app is None:
        _logger.error("PyQT4 not installed")
        sys.exit(1)
    
    try:
        from ..core.gui.ImageScreener import MainWindow as Screener
    except:
        _logger.error("Failed to load screener window")
        _logger.exception("Failed to load screener window")
        raise
        sys.exit(1)
    
    config_file = 'cfg/project.cfg'
    param = settings.parse_config_simple(config_file, coordinate_file="", pow_file="", small_micrograph_file="", selection_file="") if os.path.exists(config_file) else {}
    coordinate_file = spider_utility.spider_searchpath(param.get('coordinate_file', 'local/coords/sndc000001.*'))
    small_micrograph_file = spider_utility.spider_searchpath(param.get('small_micrograph_file', 'local/mic/mic000001.*'))
    pow_file = spider_utility.spider_searchpath(param.get('pow_file', 'local/pow/pow000001.*'))
    selection_file = param.get('selection_file', 'sel_mic.dat')
    
    dialog = Screener() 
    dialog.show()
    if dialog.is_empty():
        files = sys.argv[1:] if len(sys.argv) > 1 else []
        pow_files = glob.glob(pow_file)
        if len(pow_files) > 0:
            val = QtGui.QMessageBox.question(dialog, 'Load project files?', 'Found power spectra for screening in current project directory. Would you like to load them?', QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
            if val == QtGui.QMessageBox.Yes: files = pow_files
        dialog.openImageFiles(files)
    else:
        if len(sys.argv) > 1: _logger.info("Ignoring command line arguments - project already loaded")
    mic_files = glob.glob(small_micrograph_file)
    coord_files = glob.glob(coordinate_file)
    if len(mic_files) > 0:dialog.setAlternateImage(mic_files[0], True)
    if len(coord_files) > 0:dialog.setCoordinateFile(coord_files[0], True)
    
    sys.exit(app.exec_())


