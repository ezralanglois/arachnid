'''
.. Created on Oct 5, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import logging
from ..parallel import process_tasks
import ndimage_file
import numpy 

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
    
    img1 = ndimage_file.read_image(images[0])
    img = image_processor(img1, 0, **extra).ravel()
    mat = numpy.zeros((len(images), img.shape[0]), dtype=dtype)
    for row, data in process_tasks.for_process_mp(ndimage_file.iter_images(images), image_processor, img1.shape, queue_limit=100, **extra):
        mat[row, :] = data.ravel()[:img.shape[0]]
    return mat


def process_images(input_file, output_file, transform_func, index=None, **extra):
    ''' Apply functor to each image in the list
    '''
    
    for i, img in enumerate(ndimage_file.iter_images(input_file, index)):
        img = transform_func((i, img), **extra)
        ndimage_file.write_image(output_file, img, i)

from ..metadata import spider_utility, format_utility
import scipy.io
from ..parallel import process_queue

def read_image_mat(filename, label, image_processor, shared=False, cache_file=None, force_mat=False, dtype=numpy.float, **extra):
    '''Create a matrix where each row is an image
    
    :Parameters:
    
    filename : str
               Name of the file
    label : array
            Array of selected indicies
    image_processor : function
                      Extract features from the image 
    shared : bool
             If True create a shared memory array
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    mat : array
          2D matrix where each row is an image
    '''
    
    if not isinstance(filename, dict) and not hasattr(filename, 'find'): filename=filename[0]
    if hasattr(label, 'ndim') and label.ndim == 2:
        filename = spider_utility.spider_filename(filename, int(label[0, 0]))
        index = int(label[0, 1])
    else: index = int(label[0])
    img1 = ndimage_file.read_image(filename, index)
    img = image_processor(img1, 0, **extra).ravel()
    
    if force_mat:
        if cache_file is not None and cache_file != "":
            cache_file = format_utility.new_filename(cache_file, suffix="_read", ext=".mat")
            if format_utility.os.path.exists(cache_file):
                data = scipy.io.loadmat(cache_file)['data']
                return numpy.ascontiguousarray(data)
    else:
        if cache_file is not None and cache_file != "":
            cache_file = format_utility.new_filename(cache_file, suffix="_read", ext=".mat")
            cache_dat = format_utility.new_filename(cache_file, suffix="_read_data", ext=".bin")
            if format_utility.os.path.exists(cache_file):
                _logger.info("Reading data matrix from %s"%cache_file)
                mat = scipy.io.loadmat(cache_file)
                dtype = mat['dtype'][0]
                if dtype[0] == '[': dtype = dtype[2:len(dtype)-2]
                n = numpy.prod(mat['coo'])
                if n == (img.shape[0]*label.shape[0]):
                    if shared:
                        mat, shmem_mat = process_queue.create_global_dense_matrix( tuple(mat['coo']) )
                        mat[:] = numpy.fromfile(cache_dat, dtype=numpy.dtype(dtype))
                        return shmem_mat
                    else:
                        mat = numpy.fromfile(cache_dat, dtype=numpy.dtype(dtype)).reshape((len(label), img.shape[0]), order='C')
                        return mat
                else:
                    _logger.info("Data matrix does not match the input: %d != %d (%d*%d)"%(n, img.shape[0]*label.shape[0], img.shape[0], label.shape[0]))
    
    if shared:
        assert(False)
        mat, shmem_mat = process_queue.create_global_dense_matrix( ( len(label), img.shape[0] )  )
    else:
        mat = numpy.zeros((len(label), img.shape[0]), dtype=dtype)
        shmem_mat = mat
    for row, data in process_tasks.for_process_mp(ndimage_file.iter_images(filename, label), image_processor, img1.shape, queue_limit=100, **extra):
        mat[row, :] = data.ravel()[:img.shape[0]]
    if force_mat:
        if cache_file is not None and cache_file != "":
            scipy.io.savemat(cache_file, dict(data=mat, label=label), oned_as='column', format='5')
    else:
        if cache_file is not None and cache_file != "":
            _logger.info("Caching image matrix")
            scipy.io.savemat(cache_file, dict(coo=numpy.asarray(mat.shape, dtype=numpy.int), dtype=mat.dtype.name), oned_as='column', format='5')
            mat.ravel().tofile(cache_dat)
    return shmem_mat



