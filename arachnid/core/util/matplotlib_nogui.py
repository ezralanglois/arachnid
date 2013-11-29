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
    from ..app import tracing
    import logging
    tracing.log_import_error("Cannot import plotting libraries - plotting disabled", logging.getLogger())
    pylab=None

