''' Similairty between images

.. Created on Dec 22, 2013
.. codeauthor:: robertlanglois
'''
from ..app import tracing
import logging
import numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from util import _transform
    _transform;
except:
    tracing.log_import_error('Failed to load _transform.so module - certain functions will not be available', _logger)
    _transform=None


def normalize_correlation(ccmap, image, tshape, template_ssd):
    '''
    '''
    
    img_int = image.cumsum(1).cumsum(0)
    img_int_sq = numpy.square(image).cumsum(1).cumsum(0)
    #assert(numpy.alltrue(numpy.isfinite(ccmap)))
    #assert(numpy.alltrue(numpy.isfinite(img_int)))
    #assert(numpy.alltrue(numpy.isfinite(img_int_sq)))
    #assert(template_ssd>0)
    _transform.normalize_correlation(ccmap, img_int, img_int_sq, tshape[0], tshape[1], template_ssd)

