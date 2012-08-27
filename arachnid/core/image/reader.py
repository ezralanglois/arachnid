''' Read images from a variety of formats

The image reading interface automatically determines the image format
when reading the image. It solely depends on `EMAN2`_ to perform this
task.

The following formats are supported for reading:

============= ==============
MRC           IMAGIC
SPIDER        HDF5
PIF           ICOS
VTK           PGM
Amira         Video-4-Linux
Gatan DM2     Gatan DM3
TIFF          Scans-a-lot
LST           PNG
============= ==============

.. note::
    
    If the third-party libraries are not installed or detected, then the particular read function
    does not return an error, it is merely disabled.

When reading MRC files using EMAN, the image is flipped.

.. _`EMAN2`: http://blake.bcm.tmc.edu/eman/eman2/

.. Created on Sep 28, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def is_supported_format(filename):
    '''Test if the file has an image in a supported format
    
    :Parameters:

    filename : string
               The name and path of the file
    
    :Returns:

    return_val : bool
                 True if image format is supported
    '''
    
    try:
        return supported_format_using_eman2(filename)
    except:
        return False

def supported_format(filename):
    '''Test if the file has an image in a supported format
    
    :Parameters:

    filename : string
               The name and path of the file
    
    :Returns:

    return_val : bool
                 True if image format is supported
    '''
    
    try:
        return supported_format_using_eman2(filename)
    except:
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.exception("Failed to find file format using EMAN2")
        raise IOError, "Format not supported: %s"%filename
    
def is_mrc(filename):
    ''' Test if the file has an image in the MRC format
    
    :Parameters:

    filename : string
               The name and path of the file
    
    :Returns:

    return_val : bool
                 True if image format is supported
    '''
    
    return is_mrc_using_eman2(filename)

def read_image(filename, index=None, force_flip=False, stack_as_vol=False, emdata=None):
    '''
    Read an image using the given filename
    
    A simple interface to read an image from a file, which may contain a stack of images.
    
    .. sourcecode:: py
    
        >>> from core.image.reader import *
        >>> img0 = read_image("image_stack.spi")
        
        >>> img7 = read_image("image_stack.spi", 6)
    
    :Parameters:

    filename : str
               The name and path of the file
    index : int
            Index of the image in a stack
    stack_as_vol : bool
                   Treat a stack as a volume
    
    :Returns:

    return_val : image-like object
                 Returns the image in the library native format.
    '''
    
    try:
        if index is not None: index = int(index)
        return read_image_using_eman2(filename, index, force_flip, stack_as_vol, emdata)
    except:
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.exception("Failed to read file using EMAN2 -- %s"%filename)
        else:
            _logger.error("Failed to read file using EMAN2 -- %s"%filename)
        raise IOError, "Failed to read file: %s"%str(filename)

def count_images(filename, vol_test=False):
    '''
    Count the number of images in a stack
    
    A simple interface to count the number of images in a stack.
    
    .. sourcecode:: py
    
        >>> from core.image.reader import *
        >>> count_images("image_stack.spi")
        50
    
    :Parameters:

    filename : string
               Name and path of the file
    vol_test : bool
               Should test whether image is a volume and return number of slices
    
    :Returns:

    count : integer
            Number of images in the stack
    '''
    
    if isinstance(filename, list):
        count = 0
        for f in filename:
            count += count_images(f, vol_test)
        return count
    
    try:
        count = count_images_using_eman2(filename, vol_test)
    except:
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.exception("Failed to count images in file using EMAN2")
        else:
            _logger.error("Failed to count images in file using EMAN2")
        raise IOError, "Failed to read image header: %s"%filename
    
    return count

def read_stack(filename, sel=None, imgs=None, force_flip=False):
    '''Read a stack of images using the given filename and list of selections
    
    A simple interface to read a stack of images from a file into a list.
    
    .. sourcecode:: py
    
        >>> from core.image.reader import *
        >>> all_imgs = read_image("image_stack.spi")
        >>> len(sel_imgs)
        50
        
        >>> sel_imgs = read_image("image_stack.spi", xrange(5,10,2))
        >>> len(sel_imgs)
        3
        
        >>> read_image("image_stack.spi", xrange(15,30), sel_imgs)
        >>> len(sel_imgs)
        18
    
    :Parameters:

    filename : string
               The name and path of the file
    index : integer
            Index of the image in a stack
    
    :Returns:

    imgs : list
          List of images in the library native format.
    '''
    
    if imgs is None: imgs = []
    
    if sel is None:
        image_count = count_images(filename)
        sel = xrange(image_count)
    
    for index in sel:
        try: int(index)
        except: index = index.id-1
        imgs.append( read_image(filename, index, force_flip) )
    return imgs

try:
    import eman2_utility
    
    def supported_format_using_eman2(filename):
        ''' Test if the file has an image in a format supported by EMAN2
        
        :Parameters:

        filename : string
                   The name and path of the file
        
        :Returns:

        return_val : bool
                     True if image format is supported
        '''
        
        return eman2_utility.EMUtil.get_image_type(filename) != eman2_utility.EMUtil.ImageType.IMAGE_UNKNOWN
    
    def is_mrc_using_eman2(filename):
        ''' Test if the file has an image in the MRC format
        
        :Parameters:

        filename : string
                   The name and path of the file
        
        :Returns:

        return_val : bool
                     True if image format is supported
        '''
        
        return eman2_utility.EMUtil.get_image_type(filename) == eman2_utility.EMUtil.ImageType.IMAGE_MRC
    
    def read_image_using_eman2(filename, index, force_flip, stack_as_vol=False, emdata=None):
        '''Read an image using the given filename
        
        A simple interface to read an image from a file using EMAN2, which may contain a stack of images.
        
        :Parameters:

        filename : string
                   The name and path of the file
        index : integer
                Index of the image in a stack
        
        :Returns:
    
        img : EMAN2.EMData
              EMAN2 EMData image object
        '''
    
        try:
            type = eman2_utility.EMUtil.get_image_type(filename)
        except:
            _logger.debug("Cannot get type of "+str(filename))
            raise IOError, "Failed to get type of file: %s"%filename
        if type == eman2_utility.EMUtil.ImageType.IMAGE_UNKNOWN: raise IOError, "Cannot find EMAN2 compatible parser for image file: "+filename
        if emdata is None: emdata = eman2_utility.EMData()
        if index is None:
            emdata.read_image_c(filename)
        else:
            if stack_as_vol:
                raise ValueError, "Unsupported function"
            else:
                emdata.read_image_c(filename, index)
        
        if type == eman2_utility.EMUtil.ImageType.IMAGE_MRC:
            
            try: mrc_label = emdata.get_attr('MRC.label0')
            except: mrc_label = ""
            if force_flip or (mrc_label.find('IMAGIC') != -1 and mrc_label.find('SPIDER') != -1):
                _logger.debug("Flipping MRC")
                emdata.process_inplace("xform.flip",{"axis":"y"})
            else:
                _logger.debug("Disable MRC flipping - not IMAGIC")
        
        return emdata
    
    def count_images_using_eman2(filename, vol_test):
        '''Count the number of images in a stack
        
        A simple interface to count the number of images in a stack.
    
        :Parameters:

        filename : string
                   Name and path of the file
        vol_test : bool
                   Should test whether image is a volume and return number of slices
        
        :Returns:

        count : integer
                Number of images in the stack
        '''
        
        try:
            count = eman2_utility.EMUtil.get_image_count(filename)
            if vol_test and count == 1:
                tmp = eman2_utility.EMData()
                tmp.read_image(filename, 0, True)
                if tmp.get_attr("nz") > 1:
                    return tmp.get_attr("nz")
            return count
        except:
            return 1
        
except:
    num_handlers=len(_logger.handlers)
    if num_handlers == 0:
        _logger.addHandler(logging.StreamHandler())
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.exception("Cannot load EMAN2 - see documentation for more details")
    else:
        _logger.warn("Cannot load EMAN2 - see documentation for more details")
    if num_handlers == 0:
        _logger.removeHandler(_logger.handlers[0])


