''' Alignment using cross-correlation

.. Created on Jan 14, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
.. codeauthor:: Ryan Hyde Smith <rhs2132@columbia.edu>
'''
import numpy
import scipy.fftpack


def xcorr_dft_peak(f1, f2, usfac, search_radius, y0=0, x0=0):
    '''
    .. codeauthor:: Robert Langlois <rl2528@columbia.edu>
    .. codeauthor:: Ryan Hyde Smith <rhs2132@columbia.edu>
    '''
    
    f3 = numpy.multiply(f1, f2.conj())
    y, x, p = _xcorr_dft_peak(f3, min(2, usfac), search_radius, y0, x0, tshape, template_ssd)
    y += y0
    x += x0
    if usfac > 2:
        dy, dx, p = _xcorr_dft_peak(f3, usfac, 1.5, y, x, tshape, template_ssd)
        y += dy
        x += dx
    return numpy.asarray((y, x, p))
    
def _xcorr_dft_peak(f3, usfac, search_radius, y0=0, x0=0):
    '''
    .. codeauthor:: Ryan Hyde Smith <rhs2132@columbia.edu>
    '''
    
    ny, nx = f3.shape
    noyx = numpy.ceil(search_radius*usfac)
    dftshift = numpy.fix(numpy.ceil(search_radius*usfac)/2)
    yoff = dftshift - numpy.rint(usfac*y0)
    xoff = dftshift - numpy.rint(usfac*x0)
    kerny = numpy.exp((-2j*numpy.pi/(ny*usfac)*(numpy.arange(noyx).T - yoff)[:, numpy.newaxis])*(scipy.fftpack.ifftshift(numpy.arange(ny) - numpy.floor(ny/2)).T[numpy.newaxis, :]))
    kernx = numpy.exp((-2j*numpy.pi/(nx*usfac)*(numpy.arange(noyx).T - xoff)[:, numpy.newaxis])*(scipy.fftpack.ifftshift(numpy.arange(nx) - numpy.floor(nx/2)).T[numpy.newaxis, :])).T
    CC = numpy.dot(numpy.dot(kerny, f3), kernx)
    dy, dx = numpy.unravel_index(numpy.argmax(CC), CC.shape)
    peak=CC[dy,dx].real
    dy = (float(dy) - dftshift)/usfac
    dx = (float(dx) - dftshift)/usfac
    return dy, dx, peak

