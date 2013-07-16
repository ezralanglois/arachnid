''' Handle imports for matplotlib

.. Created on Jul 16, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

try:
    import matplotlib
    matplotlib.use('Agg')
    import pylab
    pylab;
except:
    from ..core.app import tracing
    tracing.log_import_error("Cannot import plotting libraries - plotting disabled", _logger)
    pylab=None

