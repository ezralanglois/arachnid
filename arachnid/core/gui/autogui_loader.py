''' Defines an interface for the AutoGUI

This is a workaround when the GUI segfaults so it will not take down command line programs.


.. Created on Apr 1, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import logging
import sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def display(program, display_gui=False, **extra):
    '''
    '''
    
    if not display_gui: return
    logging.getLogger().setLevel(logging.INFO)
    from AutoGUI import Dialog
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
    #if QtGui is None: return # Would be nice, but causes segfault
    from ..app.settings import OptionGroup
    if pgroup is None: pgroup=parser
    group = OptionGroup(parser, "User Interface", "Options to control the state of the AutoGUI", id=__name__)
    group.add_option("-X", display_gui=False,       help="Display the graphical user interface", gui=dict(nogui=True), dependent=False)
    pgroup.add_option_group(group)


