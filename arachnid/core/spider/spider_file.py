''' Convience functions for dealing with SPIDER files

.. Created on Oct 18, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..image import ndimage_file
import os


def copy_to_spider(filename, tempfile, index=None):
    ''' Test if input file is in SPIDER format, if not copy to tempfile
    
    :Parameters:
    
    filename : str
               Input filename to test
    tempfile : str
               Output filename (if input file not SPIDER format)
    index : int, optional
            Index of image in stack
    
    :Returns:
    
    spider_file : str
                  Name of a file containing the image in SPIDER format
    '''
    
    if is_spider_image(filename) and os.path.splitext(filename)[1] == os.path.splitext(tempfile)[1]: return filename
    
    if ndimage_file.mrc.is_readable(filename):
        img = ndimage_file.mrc.read_image(filename, index)
    else: img = ndimage_file.read_image(filename, index)
    if ndimage_file.eman_format.is_avaliable():
        ndimage_file.eman_format.write_spider_image(tempfile, img) # TODO: remove this 
    else:
        ndimage_file.spider.write_image(tempfile, img)
    return tempfile


def is_spider_image(filename):
    ''' Test if input file is in SPIDER format
    
    :Parameters:
    
    filename : str
               Input filename to test
    
    :Returns:
    
    is_spider : bool
                True if file is in SPIDER format
    '''
    
    filename = ndimage_file.readlinkabs(filename)
    return ndimage_file.spider.is_readable(filename)

