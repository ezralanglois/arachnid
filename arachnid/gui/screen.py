''' Graphical user interface for screening images

.. Created on Jul 17, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.gui.util import qtapp
from ..core.gui.util.qt4_loader import QtGui
from ..core.app import tracing
import logging, sys, glob

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
    
    dialog = Screener() 
    dialog.show()
    if dialog.is_empty():
        files = sys.argv[1:] if len(sys.argv) > 1 else []
        pow_files = glob.glob('local/pow/pow*.*')
        if len(pow_files) > 0:
            val = QtGui.QMessageBox.question(dialog, 'Load project files?', 'Found power spectra for screening in current project directory. Would you like to load them?', QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
            if val == QtGui.QMessageBox.Yes: files = pow_files
        dialog.openImageFiles(sys.argv[1:])
        """
        val = QtGui.QMessageBox.question(dialog, 'Load project files?', 'Found %s for screening in current project directory. Would you like to load them?'%msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        if len(sys.argv) > 1:
            dialog.openImageFiles(sys.argv[1:])
        else:
            pow_files = glob.glob('local/pow/pow*.*')
            if len(pow_files) > 0:
                mic_files = glob.glob('local/mic/mic*.*')
                coord_files = glob.glob('local/coords/sndc*.*')
                msg = 'power spectra'
                if len(mic_files) > 0:
                    msg += (", " if len(coord_files) > 0 else ' and ')
                    msg += 'decimate micrographs'
                if len(coord_files) > 0:
                    msg += ' and particle coordinates'
                val = QtGui.QMessageBox.question(dialog, 'Load project files?', 'Found %s for screening in current project directory. Would you like to load them?'%msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
                if val == QtGui.QMessageBox.Yes:
                    dialog.openImageFiles(pow_files)
                    if len(mic_files) > 0:dialog.setAlternateImage(mic_files[0])
                    if len(coord_files) > 0:dialog.setCoordinateFile(coord_files[0])
        """
    else:
        if len(sys.argv) > 1: _logger.info("Ignoring command line arguments - project already loaded")
    mic_files = glob.glob('local/mic/mic*.*')
    coord_files = glob.glob('local/coords/sndc*.*')
    if len(mic_files) > 0:dialog.setAlternateImage(mic_files[0], True)
    if len(coord_files) > 0:dialog.setCoordinateFile(coord_files[0], True)
    
    sys.exit(app.exec_())


