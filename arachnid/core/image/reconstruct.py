''' Reconstruct a set of 2D projections into 3D volume

Supported Reconstruction Methods

    - NN4: EMAN2/Sprax - Nearest-neighbor Interpolation in Fourier Space - (Requires eman2/sparx installation)
    - BP3F: SPIDER - Kaiser-Bessel Interpolation in Fourier Space
    - BP3N: SPIDER - Nearest-neighbor Interpolation in Fourier Space

.. Created on Aug 15, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
from ..parallel import mpi_utility, process_tasks #, process_queue
#from ..orient import transforms
#import ndimage_utility
try: import eman2_utility
except: raise
import logging, numpy #, scipy.fftpack

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG) #img = eman2_utility.ramp(img)

try: 
    from spi import _spider_reconstruct
    _spider_reconstruct;
except:
    _spider_reconstruct=None
    _logger.exception("Module failed to load")
    tracing.log_import_error('Failed to load _spider_util.so module - certain functions will not be available: ndimage_utility.ramp', _logger)

def reconstruct_bp3f_mp(gen, image_size, align, npad=2, **extra):
    '''Reconstruct a single volume with the given image generator and alignment
    file.
    
    :Parameters:
    
    gen : array generator
          Generate a sequence of images in the array format
    align : str
            Input alignment file
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    vol : array
          Reconstruction volume (IF MPI, then only to the root, otherwise None)
    '''
    
    fftvol, weight = None, None
    for v, w in process_tasks.iterate_reduce(gen, backproject_bp3f, align=align, npad=npad, image_size=image_size, **extra):
        if fftvol is None:
            fftvol = v.copy(order='F')
            weight = w.copy(order='F')
        else:
            fftvol += v
            weight += w
    
    mpi_utility.block_reduce(fftvol, **extra)
    mpi_utility.block_reduce(weight, **extra)
    if mpi_utility.is_root(**extra):
        vol = numpy.zeros((image_size, image_size, image_size), order='F', dtype=weight.dtype)
        _spider_reconstruct.finalize_bp3f(fftvol, weight, vol)#, image_size)
        return vol

def backproject_bp3f(gen, image_size, align, process_number, npad=2, **extra):
    '''
    '''
    
    try:
        pad_size = image_size*npad
        tabi = numpy.zeros(4999, dtype=numpy.float32)
        forvol = numpy.zeros((image_size+1, pad_size, pad_size), order='F', dtype=numpy.complex64)
        weight = numpy.zeros((image_size+1, pad_size, pad_size), order='F', dtype=tabi.dtype)
        
        _spider_reconstruct.setup_bp3f(tabi, pad_size)
        for i, img in gen:
            a = align[i]
            _spider_reconstruct.backproject_bp3f(img.T, forvol, weight, tabi, a[0], a[1], a[2])
    except:
        _logger.exception("Error in backproject worker")
        raise
    return forvol, weight

def reconstruct_bp3n_mp(gen, image_size, align, npad=2, **extra):
    '''Reconstruct a single volume with the given image generator and alignment
    file.
    
    :Parameters:
    
    gen : array generator
          Generate a sequence of images in the array format
    align : str
            Input alignment file
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    vol : array
          Reconstruction volume (IF MPI, then only to the root, otherwise None)
    '''
    
    fftvol, weight = None, None
    for v, w in process_tasks.iterate_reduce(gen, backproject_bp3n, align=align, npad=npad, image_size=image_size, **extra):
        if fftvol is None:
            fftvol = v.copy(order='F')
            weight = w.copy(order='F')
        else:
            fftvol += v
            weight += w
    mpi_utility.block_reduce(fftvol, **extra)
    mpi_utility.block_reduce(weight, **extra)
    if mpi_utility.is_root(**extra):
        vol = numpy.zeros((image_size, image_size, image_size), order='F')
        _spider_reconstruct.finalize_nn4f(fftvol, weight, vol)#, image_size)
        return vol

def backproject_bp3n(gen, image_size, align, process_number, npad=2, **extra):
    '''
    '''
    
    try:
        pad_size = image_size*npad
        forvol = numpy.zeros((image_size+1, pad_size, pad_size), order='F', dtype=numpy.complex64)
        weight = numpy.zeros((image_size+1, pad_size, pad_size), order='F', dtype=numpy.int32)
        
        for i, img in gen:
            a = align[i]
            _spider_reconstruct.backproject_nn4f(img.T, forvol, weight, a[0], a[1], a[2])
    except:
        _logger.exception("Error in backproject worker")
        raise
    return forvol, weight

def reconstruct_nn4_mp(gen, image_size, align, **extra):
    '''Reconstruct a single volume with the given image generator and alignment
    file.
    
    :Parameters:
    
    gen : array generator
          Generate a sequence of images in the array format
    align : str
            Input alignment file
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    vol : array
          Reconstruction volume (IF MPI, then only to the root, otherwise None)
    '''
    
    recon, fftvol, weight, image_size = eman2_utility.setup_nn4(image_size, **extra)
    for v, w in process_tasks.iterate_reduce(gen, eman2_utility.backproject_nn4_new, align=align, **extra):
        fftvol += v
        weight += w
    return eman2_utility.finalize_nn4(recon)

def reconstruct_nn4(gen, align=None, npad=2, sym='c1', weighting=1, **extra):
    '''Reconstruct a single volume with the given image generator and alignment
    file.
    
    :Parameters:
    
    gen : array generator
          Generate a sequence of images in the array format
    align : str
            Input alignment file
    npad : int
           Number of times to pad image before FFT
    sym : str
          Type of symmetry in the volume
    weighting : float
                Weight for the projections
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    vol : array
          Reconstruction volume (IF MPI, then only to the root, otherwise None)
    '''
    
    recon = eman2_utility.backproject_nn4(gen, align, npad=npad, sym=sym, weighting=weighting)
    mpi_utility.block_reduce(recon[1], **extra)
    mpi_utility.block_reduce(recon[2], **extra)
    if mpi_utility.is_root(**extra):
        return eman2_utility.finalize_nn4(recon)

def reconstruct_nn4_3(gen1, gen2, align1=None, align2=None, npad=2, sym='c1', weighting=1, **extra):
    '''Reconstruct a full volume and two half volumes with the given image generator and alignment
    file.
    
    :Parameters:
    
    gen1 : array generator
          Generate a sequence of images in the array format (for first half volume)
    gen2 : array generator
          Generate a sequence of images in the array format (for second half volume)
    align1 : str
            Input alignment file for first half volume
    align2 : str
            Input alignment file for second half volume
    npad : int
           Number of times to pad image before FFT
    sym : str
          Type of symmetry in the volume
    weighting : float
                Weight for the projections
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    vol : array
          Reconstruction full volume (IF MPI, then only to the root, otherwise None)
    vol1 : array
          Reconstruction half volume (IF MPI and node rank == 1, otherwise None)
    vol2 : array
          Reconstruction half volume (IF MPI and node rank == 2, otherwise None)
    '''
    
    root = 1 if mpi_utility.get_size(**extra) > 2 else 0
    recon = eman2_utility.backproject_nn4(gen1, align1, npad=npad, sym=sym, weighting=weighting)
    mpi_utility.block_reduce(recon[1], root=root, **extra)
    mpi_utility.block_reduce(recon[2], root=root, **extra)
    if root != 0:
        mpi_utility.send_to_root(recon[1], root, **extra)
        mpi_utility.send_to_root(recon[2], root, **extra)
    recon1 = recon if mpi_utility.get_rank(**extra)==root else None
    
    assert( mpi_utility.get_rank(**extra)!=root or recon1 is not None )
    
    root = 2 if mpi_utility.get_size(**extra) > 2 else 0
    recon = eman2_utility.backproject_nn4(gen2, align2, npad=npad, sym=sym, weighting=weighting)
    
    mpi_utility.block_reduce(recon[1], root=root, **extra)
    mpi_utility.block_reduce(recon[2], root=root, **extra)
    if root != 0:
        mpi_utility.send_to_root(recon[1], root, **extra)
        mpi_utility.send_to_root(recon[2], root, **extra)
        if mpi_utility.is_root(**extra) or mpi_utility.get_rank(**extra)==root:
            return eman2_utility.finalize_nn4(recon)
        elif recon1 is not None:
            return eman2_utility.finalize_nn4(recon1)
        return None
    else:
        vol = eman2_utility.finalize_nn4(recon1, recon, npad=npad, sym=sym, weighting=weighting)
        vol1 = eman2_utility.finalize_nn4(recon1)
        vol2 = eman2_utility.finalize_nn4(recon)
        return (vol, vol1, vol2)

