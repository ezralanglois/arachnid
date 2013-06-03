''' Reproject a 2D slice from a 3D volume

.. Created on Mar 8, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
from ..parallel import mpi_utility #, process_tasks
import logging, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from spi import _spider_reproject
    _spider_reproject;
except:
    _spider_reproject=None
    tracing.log_import_error('Failed to load _spider_reproject.so module', _logger)

def reproject_3q_mp(vol, rad, ang, out=None, thread_count=0):
    '''
    '''
    
    return reproject_mp(_spider_reproject.reproject_3q_omp, vol, rad, ang, out, thread_count)

def reproject_3q(vol, rad, ang, out=None, **extra):
    '''
    '''
    
    return reproject_mp_mpi(_spider_reproject.reproject_3q_omp, vol, rad, ang, out, **extra)

def reproject_3q_single(vol, rad, ang, out=None, **extra):
    '''
    '''
    
    if out is None:
        out = numpy.zeros((len(ang), vol.shape[0],  vol.shape[1]), dtype=vol.dtype)
    _spider_reproject.reproject_3q_omp(vol.T, out.T, ang.T, rad)
    return out

def reproject_mp_mpi(project, vol, rad, ang, out=None, thread_count=0, **extra):
    '''
    '''
    
    if out is None:
        out = numpy.zeros((len(ang), vol.shape[0],  vol.shape[1]), dtype=vol.dtype)
    vol = mpi_utility.broadcast(vol, **extra)
    slice = mpi_utility.mpi_slice(len(ang), **extra)
    reproject_mp(project, vol, rad, ang[slice], out[slice], thread_count)
    mpi_utility.gather_all(out, out[slice], **extra)
    # alternative, use itertor and broadcast the current set when reaches more than limit in memory
    return out

def reproject_mp(reproject_func, vol, rad, ang, out=None, thread_count=0):
    '''
    '''
    
    if out is None:
        out = numpy.zeros((len(ang), vol.shape[0], vol.shape[1]), dtype=vol.dtype)
    reproject_func(vol.T, out.T, ang.T, rad)
    return out

"""
def reproject_mp(reproject_func, vol, rad, ang, out=None, thread_count=0):
    '''
    '''
    
    if out is None:
        out = numpy.zeros((len(ang), vol.shape[0], vol.shape[1]), dtype=vol.dtype)
    if 1 == 1:
        reproject_func(vol, out, ang, rad)
    else:    
        for i, proj in process_tasks.process_queue.map_array_out(reproject_func, thread_count, ang, vol, rad):
            out[i] = proj
    return out
    

def reproject_3q_iterator(ang_gen, vol, rad, **extra):
    '''
    '''
    
    #ETUP_REPROJECT_3Q(VOL, RADIUS, NX, NY, NZ, NN)
    nn = _spider_reproject.setup_reproject_3q(vol, rad)
    ipcube = numpy.zeros((5, nn), dtype=numpy.int)
    proj = numpy.zeros((vol.shape[0],  vol.shape[1]), dtype=vol.dtype)
    for i, ang in enumerate(ang_gen):
        #REPROJECT_3Q(VOL, PROJ, IPCUBE, PSI, THETA, PHI, RADIUS, NX, NY, NZ, NN, DUM)
        _spider_reproject.reproject(vol, proj, ipcube, ang[0], ang[1], ang[2], rad)
        yield i, proj
"""


