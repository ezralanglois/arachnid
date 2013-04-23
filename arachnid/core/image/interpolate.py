''' Reproject a 2D slice from a 3D volume

.. Created on Mar 8, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
import logging, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: #300 2.26
    from spi import _spider_interpolate
    _spider_interpolate;
except:
    _spider_interpolate=None
    tracing.log_import_error('Failed to load _spider_interpolate.so module', _logger)

def interpolate_bilinear(vol, out):
    '''
    '''
    
    vol = numpy.asarray(vol, dtype=numpy.float32)
    if vol.ndim != 3: raise ValueError, "Interpolation of 2D projections not currently supported"
    if not hasattr(out, 'ndim'):
        if hasattr(out, '__len__'):
            out = numpy.zeros((int(out[0]), int(out[1]), int(out[2])))
        else:
            out = numpy.zeros((int(vol.shape[0]/out), int(vol.shape[1]/out), int(vol.shape[2]/out)), dtype=vol.dtype)
    _spider_interpolate.interpolate_bi_3(vol.T, out.T)
    return out

