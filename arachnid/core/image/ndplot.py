''' Plotting utilities


.. Created on Aug 13, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
try:
    import matplotlib
    try: import pycairo
    except:matplotlib.use('Agg')
    else: 
        try:
            matplotlib.use('cairo.png')
        except:
            tracing.log_import_error('Failed to set backend', _logger)
        pycairo;
except:
    tracing.log_import_error('Failed to set backend', _logger)
    
try: 
    import pylab
    pylab;
except:
    tracing.log_import_error('Failed to load matplotlib - plotting functions have been disabled', _logger)
    pylab=None
