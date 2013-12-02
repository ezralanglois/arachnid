'''
.. Created on Oct 5, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import logging
from ..parallel import process_tasks
from ..parallel import openmp
import ndimage_file
import numpy
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def create_matrix_from_file(images, image_processor, dtype=numpy.float, **extra):
    '''Create a matrix where each row is an image
    
    :Parameters:
    
    filename : str
               Name of the file
    label : array
            Array of selected indicies
    image_processor : function
                      Extract features from the image 
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    mat : array
          2D matrix where each row is an image
    '''
    
    openmp.set_thread_count(1)
    img1 = ndimage_file.read_image(images[0][0]) if isinstance(images[0], tuple) else ndimage_file.read_image(images[0])
    
    img = image_processor(img1, 0, **extra).ravel()
    total = len(images[1]) if isinstance(images, tuple) else len(images)
    mat = numpy.zeros((total, img.shape[0]), dtype=dtype)
    for row, data in process_tasks.for_process_mp(ndimage_file.iter_images(images), image_processor, img1.shape, queue_limit=100, **extra):
        mat[row, :] = data.ravel()[:img.shape[0]]
    openmp.set_thread_count(extra.get('thread_count', 1))
    return mat

def image_array_from_file(images, image_processor, dtype=numpy.float, **extra):
    '''Create a matrix where each row is an image
    
    :Parameters:
    
    filename : str
               Name of the file
    label : array
            Array of selected indicies
    image_processor : function
                      Extract features from the image 
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    mat : array
          3D matrix where each row is an image
    '''
    
    img1 = ndimage_file.read_image(images[0][0]) if isinstance(images[0], tuple) else ndimage_file.read_image(images[0])
    
    img = image_processor(img1, 0, **extra)
    total = len(images[1]) if isinstance(images, tuple) else len(images)
    mat = numpy.zeros((total, img.shape[0], img.shape[1]), dtype=dtype)
    for row, data in process_tasks.for_process_mp(ndimage_file.iter_images(images), image_processor, img1.shape, queue_limit=100, **extra):
        mat[row, :] = data
    return mat

_cache_header = numpy.dtype([('magic', 'S10'), ('dtype', 'S3'), ('byte_num', numpy.int16), ('ndim', numpy.int32), ])

def read_matrix_from_cache(cache_file):
    ''' Read a cached matrix from a file
    
    The cache file has the following format:
    
        #. Magic number (S10)
        #. Data type (S3)
        #. Byte count (int16)
        #. Number of dimensions (int32)
        #. Variable length shape defined by number of dimesions (int)
        #. Data
    
    :Parameters:
    
    cache_file : str
                 Filename for cached matrix
                 
    :Returns:
    
    mat : array
          Cached matrix or None
    '''
    
    if not os.path.exists(cache_file): return None
    fin = open(cache_file, 'rb')
    try:
        h = numpy.fromfile(fin, dtype=_cache_header, count=1)
        if h['magic'] != 'CACHEFORM': raise ValueError, "Cache file not in proper format"
        dtype = numpy.dtype(h['dtype'][0]+str(h['byte_num'][0]))
        ndim = h['ndim'][0]
        shape = tuple(numpy.fromfile(fin, dtype=numpy.int, count=ndim))
        return numpy.fromfile(fin, dtype=dtype, count=numpy.prod(shape)).reshape(shape, order='C')
    finally:
        fin.close()

def write_matrix_to_cache(cache_file, mat):
    ''' Cache a matrix to a file
    
    The cache file has the following format:
    
        #. Magic number (S10)
        #. Data type (S3)
        #. Byte count (int16)
        #. Number of dimensions (int32)
        #. Variable length shape defined by number of dimesions (int)
        #. Data
    
    :Parameters:
    
    cache_file : str
                 Filename for cached matrix
    mat : array
          Matrix to cache
    '''
    
    h = numpy.zeros(1, _cache_header)
    h['magic'] = 'CACHEFORM'
    h['dtype']=mat.dtype.str[:2]
    h['byte_num']=int(mat.dtype.str[2:])
    h['ndim']=mat.ndim
    fout = open(cache_file, 'wb')
    try:
        h.tofile(fout)
        numpy.asarray(mat.shape, dtype=numpy.int).tofile(fout)
        mat.ravel().tofile(fout)
    finally:
        fout.close()

def process_images(input_file, output_file, transform_func, index=None, **extra):
    ''' Apply functor to each image in the list
    '''
    
    for i, img in enumerate(ndimage_file.iter_images(input_file, index)):
        img = transform_func((i, img), **extra)
        ndimage_file.write_image(output_file, img, i)

