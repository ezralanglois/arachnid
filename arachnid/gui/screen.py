''' Graphical user interface for screening images

.. Created on Jul 17, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.gui.util import qtapp
from ..core.app import tracing
import logging
import sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def main():
    # Create GUI and display
    
    tracing.configure_logging(log_file='screen.log')
    if len(sys.argv) > 1 and sys.argv[1]=='--help':
        print "Go to http://www.arachnid.us for the help documentation"
        sys.exit(0)
    from ..core.gui.util import qtapp
    app = qtapp.create_app()
    if app is None:
        _logger.error("PyQT4 not installed")
        sys.exit(1)
    
    try:
        from ..core.gui import ImageScreener
    except:
        _logger.error("Failed to load screener window")
        _logger.exception("Failed to load screener window")
        raise
        sys.exit(1)
    
    dialog=ImageScreener.launch()
    dialog;
    sys.exit(app.exec_())


