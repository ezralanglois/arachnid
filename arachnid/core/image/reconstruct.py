''' Reconstruct a set of projections into 3D volume

.. todo:: make multi-core

.. Created on Aug 15, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..parallel import mpi_utility
try: import eman2_utility
except: raise
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

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
        vol = eman2_utility.finalize_nn4(recon1, recon)
        vol1 = eman2_utility.finalize_nn4(recon1)
        vol2 = eman2_utility.finalize_nn4(recon)
        return (vol, vol1, vol2)

