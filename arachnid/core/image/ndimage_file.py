''' Reading and writing NumPY arrays in various image formats

Supported formats:
    
     - :py:mod:`EMAN2/SPARX <formats.eman_format>`

.. Created on Aug 11, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import logging
from ..app import tracing
from formats import spider, eman_format as spider_writer
#import numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def is_spider_format(filename):
    ''' Test if input file is in SPIDER format
    
    :Parameters:
    
    filename : str
               Input filename to test
    
    :Returns:
    
    is_spider : bool
                True if file is in SPIDER format
    '''
    
    return spider.is_readable(filename)

def copy_to_spider(filename, tempfile):
    ''' Test if input file is in SPIDER format, if not copy to tempfile
    
    :Parameters:
    
    filename : str
               Input filename to test
    tempfile : str
               Output filename (if input file not SPIDER format)
    
    :Returns:
    
    spider_file : str
                  Name of a file containing the image in SPIDER format
    '''
    
    if is_spider_format(filename): return filename
    for index, img in enumerate(iter_images(filename)):
        spider_writer.write_image(tempfile, img, index)
    return tempfile

def is_readable(filename):
    ''' Test if the input filename of the image is in a recognized
    format.
    
    :Parameters:
    
    filename : str
               Input filename to test
    
    :Returns:
    
    read : bool
           True if the format is recognized
    '''
    
    return get_read_format(filename) is not None

def read_header(filename, index=None):
    '''Read the header of an image from the given file
    
    :Parameters:
    
    filename : str
               Input filename to read
    index : int, optional
            Index of image to get, if None, first image (Default: None)
    
    :Returns:
        
    out : array
          Array with header information in the file
    '''
    
    format = get_read_format(filename)
    return format.read_header(filename, index)

def read_image(filename, index=None):
    '''Read an image from the given file
    
    :Parameters:
    
    filename : str
               Input filename to read
    index : int, optional
            Index of image to get, if None, first image (Default: None)
    
    :Returns:
        
    out : array
          Array with header information in the file
    '''
    
    format = get_read_format(filename)
    return format.read_image(filename, index)

def iter_images(filename, index=None):
    ''' Read a set of images from the given file
    
    :Parameters:
    
    filename : str
               Input filename to read
    index : int, optional
            Index of image to start, if None, start with the first image (Default: None)
    
    :Returns:
        
    out : array
          Array with header information in the file
    '''
    
    format = get_read_format(filename)
    return format.iter_images(filename, index)

def count_images(filename):
    ''' Count the number of images in the file
    
    :Parameters:
    
    filename : str
               Input filename to read
    
    :Returns:
        
    out : int
          Number of images in the file
    '''
    
    if isinstance(filename, list):
        format = get_read_format(filename)
        total = 0
        for f in filename:
            total += format.count_images(f)
        return total
    format = get_read_format(filename)
    return format.count_images(filename)

def is_writable(filename):
    ''' Test if the image extension of the given filename is understood
    as a writable format.
    
    :Parameters:
    
    filename : str
               Output filename to test
    
    :Returns:
    
    write : bool
            True if the format is recognized
    '''
    
    return get_write_format(filename) is not None

def write_image(filename, img, index=None):
    ''' Write the given image to the given filename using a format
    based on the file extension, or given type.
    
    :Parameters:
    
    filename : str
               Output filename for the image
    img : array
          Image data to write out
    index : int, optional
            Index image should be written to in the stack
    format : eman2_utility.EMUtil.ImageType, optional
             Format to write image in
    '''
    
    format = get_write_format(filename)
    format.write_image(filename, img, index)
    
def get_write_format(filename):
    ''' Get the write format for the image
    
    :Parameters:
    
    filename : str
               Output filename to test
    
    :Returns:
    
    write : format
            Write format for given file extension
    '''
    
    for f in _formats:
        if f.is_writable(filename): return f
    return _default_write_format

def get_read_format(filename):
    ''' Get the write format for the image
    
    :Parameters:
    
    filename : str
               Input file to test
    
    :Returns:
    
    write : format
            Read format for given file
    '''
    
    for f in _formats:
        if f.is_readable(filename): return f
    return None


def _load():
    ''' Import available formats
    '''
    
    formats = []
    try: from formats import eman_format
    except: tracing.log_import_error("Cannot load EMAN2 - supported image formats will not be available - see documentation for more details")
    else: formats.append(eman_format)
    if len(formats) == 0: raise ImportError, "No image format modules loaded!"
    return formats, eman_format

_formats, _default_write_format = _load()

