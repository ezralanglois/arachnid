''' Plotting utilities


.. Created on Aug 13, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app.tracing import log_import_error
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
try: 
    import pylab
    pylab;
except:
    log_import_error('Failed to load matplotlib - plotting functions have been disabled', _logger)
    pylab=None
