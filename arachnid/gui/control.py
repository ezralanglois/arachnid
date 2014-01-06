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
    
    tracing.configure_logging(log_level=2, log_file='control.log')
    app = qtapp.create_app()
    if app is None:
        _logger.error("PyQT4 not installed")
        sys.exit(1)
    
    try:
        from ..core.gui.ProjectUI import MainWindow as ProjectUI
    except:
        _logger.error("Failed to load ProjectUI window")
        _logger.exception("Failed to load ProjectUI window")
        raise
        sys.exit(1)
    
    screen_shot_file=None
    if len(sys.argv) > 1 and sys.argv[1]=='--screen-shot':
        screen_shot_file=sys.argv[2]
    dialog = ProjectUI(screen_shot_file) 
    dialog.show()
        
    sys.exit(app.exec_())