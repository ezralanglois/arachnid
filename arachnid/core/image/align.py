''' Rotate a 2D projection

.. Created on Mar 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
import logging, numpy


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from spi import _spider_align
    _spider_align;
except:
    _spider_align=None
    _logger.addHandler(logging.StreamHandler())
    _logger.exception("Module failed to load")
    try:
        import _spider_align
    except:
        _logger.exception("Module failed to load")
    tracing.log_import_error('Failed to load _spider_align.so module', _logger)

def align_translation(img, ref, trang=[], imgbuf=None, refbuf=None, pad=2):
    '''
    '''
    
    rang = [0, 0, 0]
    for i in xrange(3): 
        if i < len(trang): rang[i]=trang[i]
        if rang[i] == 0: rang[i]=img.shape[i]/2 if i < img.ndim else 0
        elif rang[i] < 0: rang[i]=0
    
    nx = img.shape[0]*pad
    ny = img.shape[1]*pad
    nz = img.shape[2]*pad if img.ndim > 2 else 0
    ne = nx + 2 - nx%2
    pshape = (ne, ny, nz) if nz > 0 else (ne, ny)
    if imgbuf is None: imgbuf = numpy.zeros(pshape, dtype=numpy.float32, order='F')
    if refbuf is None: refbuf = numpy.zeros(pshape, dtype=numpy.float32, order='F')
    if img.ndim==2:
        imgbuf[:img.shape[0], :img.shape[1]]=img
        imgbuf[:ref.shape[0], :ref.shape[1]]=ref
        imgbuf[img.shape[0]:, img.shape[1]:]=img.mean()
        refbuf[ref.shape[0]:, ref.shape[1]:]=ref.mean()
    else:
        imgbuf[:img.shape[0], :img.shape[1], :img.shape[2]]=img
        imgbuf[:ref.shape[0], :ref.shape[1], :ref.shape[2]]=ref
        imgbuf[img.shape[0]:, img.shape[1]:, img.shape[2]:]=img.mean()
        refbuf[ref.shape[0]:, ref.shape[1]:, ref.shape[2]:]=ref.mean()
    
    assert(rang[0]>0)
    assert(rang[1]>0)
    assert(rang[2]==0)
    xnew, ynew, znew, peakv, flag=_spider_align.align_t(imgbuf, refbuf, img.shape[0], rang[0], rang[1], rang[2])
    if flag != 0: raise ValueError, "translation failed with code: %d"%flag
    return xnew, ynew, znew, peakv
    
