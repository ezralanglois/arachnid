''' Utilities to manipulate OpenMP functions

.. Created on Sep 5, 2012
.. codeauthor:: robertlanglois
'''
from ..app import tracing
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try:
    from core import _omp
    _omp;
except:
    if _logger.isEnabledFor(logging.DEBUG):
        tracing.log_import_error("Failed to import OpenMP module - certain functionality will not be available", _logger)
    try:
        from core import _omp
    except:
        tracing.log_import_error("Failed to import OpenMP module - certain functionality will not be available", _logger)

def set_thread_count(thread_count):
    ''' Set the number of threads to be used by OpenMP
    
    :Parameters:
    
    thread_count : int
                   Number of threads to be used by OpenMP
    '''
    
    _omp.set_num_threads(thread_count)

def get_max_threads():
    '''Get maximum number of available threads.
    Return 1 if OpenMP is disabled.
    
    :Returns:
    
    num : int
          Number of available threads
    '''
    
    return _omp.get_max_threads()
