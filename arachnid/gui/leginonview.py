'''
.. Created on Dec 4, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.gui.util import qtapp
from ..core.app import tracing
import logging, sys


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def main():
    # Create GUI and display
    
    tracing.configure_logging(log_level=2, log_file='leginon.log')
    app = qtapp.create_app()
    if app is None:
        _logger.error("PyQT4 not installed")
        sys.exit(1)
    
    try:
        from ..core.gui.LeginonUI import Widget as LeginonUI
    except:
        _logger.error("Failed to load LeginonUI window")
        _logger.exception("Failed to load LeginonUI window")
        raise
        sys.exit(1)
    
    if len(sys.argv) > 1 and sys.argv[1]=='--help':
        print "Go to http://www.arachnid.us for the help documentation"
        sys.exit(0)
    
    dialog = LeginonUI() 
    dialog.show()
    sys.exit(app.exec_())


