'''
.. Created on Jan 13, 2014
.. codeauthor:: robertlanglois
'''

import logging
import numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from core import _fastdot
    _fastdot;
except:
    from ..app import tracing
    tracing.log_import_error('Failed to load _fastdot.so module - certain functions will not be available', _logger)
    _fastdot = None

def fastdot_t2(s1, s2_t, out=None, alpha=1.0, beta=0.0):
    '''
    '''
    
    if s1.shape[1] != s2_t.shape[1]: raise ValueError, "Number of columns in each matrix does not match: %d != %d"%(s1.shape[1], s2_t.shape[1])
    if out is None: out = numpy.zeros((s1.shape[0], s2_t.shape[0]), dtype=s1.dtype)
    _fastdot.gemm(s1, s2_t, out, float(alpha), float(beta))
    return out

def fastdot_t1(s1_t, s2, out=None, alpha=1.0, beta=0.0):
    '''
    '''
    
    if s1_t.shape[0] != s2.shape[0]: raise ValueError, "Number of columns in each matrix does not match: %d != %d"%(s1_t.shape[1], s2.shape[1])
    if out is None: out = numpy.zeros((s1_t.shape[1], s2.shape[1]), dtype=s1_t.dtype)
    _fastdot.gemm_t1(s1_t, s2, out, float(alpha), float(beta))
    return out

