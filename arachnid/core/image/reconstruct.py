''' Reconstruct a set of 2D projections into 3D volume

Supported Reconstruction Methods

    - NN4: EMAN2/Sprax - Nearest-neighbor Interpolation in Fourier Space - (Requires eman2/sparx installation)
    - BP3F: SPIDER - Kaiser-Bessel Interpolation in Fourier Space
    - BP3N: SPIDER - Nearest-neighbor Interpolation in Fourier Space

.. Created on Aug 15, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
from ..parallel import mpi_utility, process_tasks
import logging, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from spi import _spider_reconstruct
    _spider_reconstruct;
except:
    _spider_reconstruct=None
    tracing.log_import_error('Failed to load _spider_reconstruct.so', _logger)

def reconstruct3_bp3f_mp(image_size, gen1, gen2, align1=None, align2=None, **extra):
    '''Reconstruct three volumes using BP3F
    
    :Parameters:
    
    image_size : int
                 Image size
    gen1 : array generator
           Generate a sequence of images in the array format
    gen2 : array generator
           Generate a sequence of images in the array format
    align1 : str
             Input alignment file
    align2 : str
             Input alignment file
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    vol : array
          Reconstruction volume (IF MPI, then only to the root, otherwise None)
    vol1 : array
          Reconstruction half volume (IF MPI, then only to the root, otherwise None)
    vol2 : array
          Reconstruction half volume (IF MPI, then only to the root, otherwise None)
    '''
    
    return reconstruct3_mp(backproject_bp3f, finalize_bp3f, image_size, gen1, gen2, align1, align2, **extra)

def reconstruct_bp3f_mp(gen, image_size, align, npad=2, cleanup_fft=True, **extra):
    '''Reconstruct a single volume with the given image generator and alignment
    file.
    
    :Parameters:
    
    gen : array generator
          Generate a sequence of images in the array format
    image_size : int
                 Image size
    align : str
            Input alignment file
    npad : int
           Number of times to pad volume
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    vol : array
          Reconstruction volume (IF MPI, then only to the root, otherwise None)
    '''
    
    fftvol, weight = reconstruct_fft(backproject_bp3f, gen, image_size, align, npad, **extra)
    if mpi_utility.is_root(**extra): return finalize_bp3f(fftvol, weight, image_size, cleanup_fft)

def finalize_bp3f(fftvol, weight, image_size, cleanup_fft):
    '''
    '''
    
    if fftvol is not None:
        vol = numpy.zeros((image_size, image_size, image_size), order='F', dtype=weight.dtype)
        #_logger.error("finalize_bp3f-1")
        _spider_reconstruct.finalize_bp3f(fftvol, weight, vol)#, image_size)
        if cleanup_fft: _spider_reconstruct.cleanup_bp3f()
        #_logger.error("finalize_bp3f-2")
        return vol
    if cleanup_fft: _spider_reconstruct.cleanup_bp3f()
    
def backproject_bp3f(gen, image_size, align, process_number, npad=2, process_image=None, psi='psi', theta='theta', phi='phi', **extra):
    '''
    '''
    
    try:
        try:
            pad_size = image_size*npad
        except:
            _logger.error("image_size: %s | npad: %s"%(str(image_size), str(npad)))
            raise
        tabi = numpy.zeros(4999, dtype=numpy.float32)
        forvol = numpy.zeros((image_size+1, pad_size, pad_size), order='F', dtype=numpy.complex64)
        weight = numpy.zeros((image_size+1, pad_size, pad_size), order='F', dtype=tabi.dtype)
        
        _spider_reconstruct.setup_bp3f(tabi, pad_size)
        if len(align) > 0 and hasattr(align[0], psi):
            for i, img in gen:
                a = align[i]
                if process_image is not None: img = process_image(img, a, **extra)
                _spider_reconstruct.backproject_bp3f(img.T, forvol, weight, tabi, getattr(a, psi), getattr(a, theta), getattr(a, phi))
        else:
            for i, img in gen:
                a = align[i]
                if process_image is not None: img = process_image(img, a, **extra)
                _spider_reconstruct.backproject_bp3f(img.T, forvol, weight, tabi, a[0], a[1], a[2])
    except:
        _logger.exception("Error in backproject worker")
        raise
    return forvol, weight

def reconstruct3_bp3n_mp(image_size, gen1, gen2, align1=None, align2=None, **extra):
    '''Reconstruct three volumes using BP3F
    
    :Parameters:
    
    image_size : int
                 Image size
    gen1 : array generator
           Generate a sequence of images in the array format
    gen2 : array generator
           Generate a sequence of images in the array format
    align1 : str
             Input alignment file
    align2 : str
             Input alignment file
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    vol : array
          Reconstruction volume (IF MPI, then only to the root, otherwise None)
    vol1 : array
          Reconstruction half volume (IF MPI, then only to the root, otherwise None)
    vol2 : array
          Reconstruction half volume (IF MPI, then only to the root, otherwise None)
    '''
    
    return reconstruct3_mp(backproject_bp3n, finalize_bp3n, image_size, gen1, gen2, align1, align2, **extra)

def reconstruct_bp3n_mp(gen, image_size, align, npad=2, cleanup_fft=True, **extra):
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
    
    
    fftvol, weight = reconstruct_fft(backproject_bp3n, gen, image_size, align, npad, **extra)
    if mpi_utility.is_root(**extra): return finalize_bp3n(fftvol, weight, image_size, cleanup_fft)
    
def finalize_bp3n(fftvol, weight, image_size, cleanup_fft):
    '''
    '''
    
    if fftvol is not None:
        vol = numpy.zeros((image_size, image_size, image_size), order='F', dtype=weight.dtype)
        _spider_reconstruct.finalize_nn4f(fftvol, weight, vol)#, image_size)
        if cleanup_fft: _spider_reconstruct.cleanup_nn4f()
        return vol
    if cleanup_fft: _spider_reconstruct.cleanup_nn4f()

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

def reconstruct3_mp(backproject, finalize, image_size, gen1, gen2, align1=None, align2=None, npad=2, cleanup_fft=True, **extra):
    '''Reconstruct three volumes using BP3F
    
    :Parameters:
    
    image_size : int
                 Image size
    gen1 : array generator
           Generate a sequence of images in the array format
    gen2 : array generator
           Generate a sequence of images in the array format
    align1 : str
             Input alignment file
    align2 : str
             Input alignment file
    npad : int
           Number of times to pad volume
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    vol : array
          Reconstruction volume (IF MPI, then only to the root, otherwise None)
    vol1 : array
          Reconstruction half volume (IF MPI, then only to the root, otherwise None)
    vol2 : array
          Reconstruction half volume (IF MPI, then only to the root, otherwise None)
    '''
    
    if mpi_utility.get_size(**extra) > 100 and 1 == 0:
        rank = mpi_utility.get_rank(**extra)
        recon = reconstruct_fft(backproject, gen1, image_size, align1, npad, **extra)#, root=1
        #mpi_utility.send_to_root(recon[0], 1, **extra)
        #mpi_utility.send_to_root(recon[1], 1, **extra)
        recon1 = reconstruct_fft(backproject, gen2, image_size, align2, npad, **extra)#, root=2
        #mpi_utility.send_to_root(recon1[0], 2, **extra)
        #mpi_utility.send_to_root(recon1[1], 2, **extra)
        hvol1, hvol2 = None, None
        if rank == 1:
            hvol1 = finalize(recon[0], recon[1], image_size, cleanup_fft)
        elif rank == 2:
            hvol2 = finalize(recon1[0], recon1[1], image_size, cleanup_fft)
        elif rank == 0:
            recon[0]+=recon1[0]
            recon[1]+=recon1[1]
            del recon1
            vol = finalize(recon[0], recon[1], image_size, cleanup_fft)
            hvol1 = vol.copy(order='F')
            hvol2 = vol.copy(order='F')
        else:
            finalize(None, None, 0)
        mpi_utility.send_to_root(hvol1, 1, **extra)
        mpi_utility.send_to_root(hvol2, 2, **extra)
    else:
        _logger.info("Started back projection of %d even projections with %d threads on node %s"%(len(align1), extra['thread_count'], mpi_utility.hostname()))
        recon = reconstruct_fft(backproject, gen1, image_size, align1, npad, **extra)
        _logger.info("Started back projection of %d odd projections with %d threads on node %s"%(len(align1), extra['thread_count'], mpi_utility.hostname()))
        recon1 = reconstruct_fft(backproject, gen2, image_size, align2, npad, **extra)
        
        if mpi_utility.is_root(**extra):
            #_logger.error("finalize-1-full")
            vol = finalize(recon[0]+recon1[0], recon[1]+recon1[1], image_size, cleanup_fft)
            #_logger.error("finalize-2-half1")
            hvol1 = finalize(recon[0], recon[1], image_size, cleanup_fft)
            #_logger.error("finalize-3-half2")
            hvol2 = finalize(recon1[0], recon1[1], image_size, cleanup_fft)
            #_logger.error("finalize-3-half2-done")
            return (vol, hvol1, hvol2)
        else:
            finalize(None, None, 0, cleanup_fft)
        return None

def reconstruct_fft(backproject, gen, image_size, align, npad=2, **extra):
    '''Reconstruct a single volume with the given image generator and alignment file.
    
    :Parameters:
    
    gen : array generator
          Generate a sequence of images in the array format
    image_size : int
                 Image size
    align : str
            Input alignment file
    npad : int
           Number of times to pad volume
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    fftvol : array
             Fourier volume
    weight : array
             Weight volume
    '''
    
    fftvol, weight = None, None
    for v, w in process_tasks.iterate_reduce(gen, backproject, align=align, npad=npad, image_size=image_size, **extra):
        if fftvol is None:
            fftvol = v
            weight = w
        else:
            fftvol += v
            weight += w
    assert(fftvol is not None)
    assert(weight is not None)
    #_logger.info("begin-block_reduce1: %f"%numpy.sum(fftvol.real))
    mpi_utility.block_reduce(fftvol.ravel(order='F'), **extra)
    #_logger.info("begin-block_reduce2: %f -- %f"%(numpy.sum(fftvol.real), numpy.sum(tmp.real)))
    mpi_utility.block_reduce(weight.ravel(order='F'), **extra)
    return fftvol, weight

