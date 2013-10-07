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

