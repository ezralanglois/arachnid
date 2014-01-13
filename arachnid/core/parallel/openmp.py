''' Utilities to manipulate OpenMP functions

.. Created on Sep 5, 2012
.. codeauthor:: robertlanglois
'''

import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try:
    from core import _omp
    _omp;
except:
    from ..app import tracing
    tracing.log_import_error("Failed to import OpenMP module - certain functionality will not be available", _logger)
    _omp = None

try: 
    import mkl
    mkl;
except:
    mkl=None
    
def is_openmp_enabled():
    ''' Test if OpenMP is enabled
    
    :Returns:
    
    out : bool
          True if openmp is enabled
    '''
    
    if _omp is not None:
        return _omp.get_max_threads() > 0
    return False

def set_thread_count(thread_count):
    ''' Set the number of threads to be used by OpenMP
    
    :Parameters:
    
    thread_count : int
                   Number of threads to be used by OpenMP
    '''
    
    if mkl is not None: mkl.set_num_threads(thread_count)
    _omp.set_num_threads(thread_count)

def get_max_threads():
    '''Get maximum number of available threads.
    Return 1 if OpenMP is disabled.
    
    :Returns:
    
    num : int
          Number of available threads
    '''
    
    if _omp is not None:
        return _omp.get_max_threads()
    return 1

def get_num_procs():
    '''
    '''
    
    if _omp is not None:
        return _omp.get_num_procs()
    return 1
    
