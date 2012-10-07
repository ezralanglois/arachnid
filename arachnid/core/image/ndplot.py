''' Plotting utilities


.. Created on Aug 13, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
import logging, sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
try:
    if 'matplotlib' not in set(sys.modules.keys()):
        import matplotlib
        try: import pycairo
        except:
            if matplotlib.get_backend() != "Agg": matplotlib.use('Agg')
        else: 
            try:
                if matplotlib.get_backend() != "cairo.png": matplotlib.use('cairo.png')
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