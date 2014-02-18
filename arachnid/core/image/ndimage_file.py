''' Reading and writing NumPY arrays in various image formats

.. beg-dev

This module supports the standard formats for image data in cryo-EM, which
include:
    
     - :py:mod:`MRC <formats.mrc>`
     - :py:mod:`SPIDER <formats.spider>`
     - :py:mod:`EMAN2/SPARX <formats.eman_format>` (Optional, used if available)

.. note:: 
    
    Each read function supports soft links or shortcuts

Images are read in as standard NumPy arrays using C-ordering for 2D
or 3D images.

.. sourcecode:: py
    
    from arachnid.core.image import ndimage_file as imfile
    image = imfile.read_image('image.spi')
    
    print 'Mean pixel:', image.mean()
    print 'Image size (rows, columns):', image.shape

Due to the sheer number, images in cryo-EM are often grouped into stacks. The
number of images in a stack can be determined by:

.. sourcecode:: py

    print "Number of images:", imfile.count_images('stack.spi')

A NumPy array containing pixel values can be written out as an image
using the following:

.. sourcecode:: py

    imfile.write_image('image.mrc', image)

.. end-dev

.. Created on Aug 11, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import logging, os
from formats import spider, mrc, eman_format
from ..metadata import spider_utility, format_utility
from ..parallel import mpi_utility
import numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def copy_local(filename, selection, local_file, **extra):
    ''' Copy a stack or set of stacks to a single stack on a remote drive.
    MPI only
    
    :Parameters:
    
    filename : str
               Input filename template
    selection : array
                Selection ids
    local_file : str
                 Output filename template
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    local_file : str
                 Local filename based on rank of node and process id
    '''
    
    if mpi_utility.get_size(**extra) < 2: return filename
    
    local_file = mpi_utility.safe_tempfile(local_file, shmem=False, **extra)
    selection_file = format_utility.add_prefix(local_file, 'sel_')
    if os.path.exists(local_file) and os.path.exists(selection_file):
        remote_select = numpy.loadtxt(selection_file, delimiter=",")
        if remote_select.shape[0]==selection.shape[0] and numpy.alltrue(remote_select==selection) and count_images(local_file) == selection.shape[0]: return local_file
    
    for i, img in enumerate(iter_images(filename, selection)):
        _logger.debug("Caching: %s - %d@%s"%(str(selection[i]), i, local_file))
        write_image(local_file, img, i)
    numpy.savetxt(selection_file, selection, delimiter=",")
    return local_file

def is_readable(filename):
    ''' Test if the input filename of the image is in a recognized
    format.
    
    :Parameters:
    
    filename : str
               Input filename to test
    
    :Returns:
    
    out : bool
           True if the format is recognized
    '''
    
    filename = readlinkabs(filename)
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
    
    filename = readlinkabs(filename)
    format = get_read_format_except(filename)
    return format.read_header(filename, index)

def read_image(filename, index=None, **extra):
    '''Read an image from the given file
    
    :Parameters:
    
    filename : str
               Input filename to read
    index : int, optional
            Index of image to get, if None, first image (Default: None)
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    out : array
          Array with header information in the file
    '''
    
    if isinstance(filename, tuple): filename,index=filename
    filename = readlinkabs(filename)
    format = get_read_format_except(filename)
    cache_keys = format.cache_data().keys()
    param = {}
    for key in cache_keys:
        if key not in extra: continue
        param[key] = extra[key]
    try:
        return format.read_image(filename, index, **param)
    except:
        if index is not None:
            _logger.error("Error reading: %d@%s"%(index, filename))
        else:
            _logger.error("Error reading: %s"%filename)
        raise

def read_stack(filename):
    ''' Read an entire stack into a multi-dimensional array
    
    :Parameters:
    
    filename : str
               Input filename to read
    
    :Returns:
        
    out : array
          Array nxm1xm2xm3 where n is the 
          number of images and m1-m3 are
          the dimensions of an individual
          image
    '''
    
    img = read_image(filename)
    count = count_images(filename)
    stack = numpy.zeros((count, )+img.shape)
    for i, img in enumerate(iter_images(filename)):
        stack[i, :] = img
    return stack

def iter_images(filename, index=None):
    ''' Read a set of images from the given file
    
    :Parameters:
    
    filename : str
               Input filename to read
    index : int or array, optional
            Image index: 
                - if int, then index of image to start 
                - if array, then an array of selected images
                - if list of strs, then iterate over each stack for each filename in list
                - if list of tuples, then assumes each is a filename and an index
                - if None, start with the first image (Default: None)
    
    :Returns:
        
    out : array
          Array with header information in the file
        
    .. todo:: iter single images
    '''
    
    if isinstance(filename, tuple): filename, index = filename
    if index is None and isinstance(filename, list):
        if isinstance(filename[0], tuple):
            for f, id in filename:
                yield read_image(f, id-1)
        else:
            for f in filename:
                for img in iter_images(f):
                    yield img
        return
    elif index is not None and hasattr(index, 'ndim'):
        if hasattr(filename, 'find') and count_images(filename) == 1:
            for i in xrange(len(index)):
                if index.ndim == 2:
                    curr_filename = spider_utility.spider_filename(filename, int(index[i, 0]))
                else:
                    curr_filename = spider_utility.spider_filename(filename, int(index[i]))
                yield read_image(curr_filename)
            return
        elif index.ndim == 2 and index.shape[1]>1:
            if index[:, 1].min() < 0: raise ValueError, "Cannot have a negative index"
            
            beg = 0
            tot = len(numpy.unique(index[:, 0].astype(numpy.int)))
            if not isinstance(filename, dict) and not hasattr(filename, 'find'): filename=filename[0]
            for i in xrange(tot):
                id = index[beg, 0]
                curr_filename = spider_utility.spider_filename(filename, int(id)) if not isinstance(filename, dict) else filename[int(id)]
                sel = numpy.argwhere(id == index[:, 0]).ravel()
                if beg != sel[0]: raise ValueError, "Array must be sorted by file ids: %d != %d -- %f, %f"%((beg), sel[0], index[beg, 0], beg)
                try:
                    curr_filename = readlinkabs(curr_filename)
                    if index[sel, 1].min() < 0: raise ValueError, "Cannot have a negative index"
                    if numpy.any(index[sel, 1]) > count_images(curr_filename):
                        raise ValueError, "Index exceeds stack size: %s - %d > %d"%(curr_filename, index[sel, 1].max(), count_images(curr_filename))
                    if len(sel) > 1:
                        for img in iter_images(curr_filename, index[sel, 1]):
                            yield img
                    else:
                        yield read_image(curr_filename, int(index[sel[0], 1]))
                except:
                    _logger.error("stack filename: %s - %d to %d"%(curr_filename, numpy.min(index[sel, 1]), numpy.max(index[sel, 1])))
                    raise
                beg += sel.shape[0]
            return
        
        
    if index is not None and hasattr(index, '__iter__') and not hasattr(index, 'ndim'): index = numpy.asarray(index)
    filename = readlinkabs(filename)
    format = get_read_format_except(filename)
    for img in format.iter_images(filename, index):
        yield img

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
        format = get_read_format_except(readlinkabs(filename[0]))
        total = 0
        for f in filename:
            total += format.count_images(f)
        return total
    else:
        filename = readlinkabs(filename)
        format = get_read_format_except(filename)
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

def write_image(filename, img, index=None, header=None, inplace=False):
    ''' Write the given image to the given filename using a format
    based on the file extension, or given type.
    
    :Parameters:
    
    filename : str
               Output filename for the image
    img : array
          Image data to write out
    index : int, optional
            Index image should be written to in the stack
    header : dict
            Header dictionary
    inplace : bool
              Write new image to stack without removing the stack
    '''
    
    
    format = get_write_format(filename)
    if format is None: 
        raise IOError, "Could not find format for extension of %s"%filename
    format.write_image(filename, img, index, header, inplace)
    
def write_stack(filename, imgs):
    ''' Write the given image to the given filename using a format
    based on the file extension, or given type.
    
    :Parameters:
    
    filename : str
               Output filename for the image
    imgs : array
           Image stack data to write out
    '''
    
    format = get_write_format(filename)
    if format is None: 
        raise IOError, "Could not find format for extension of %s"%filename
    index = 0
    for img in imgs:
        format.write_image(filename, img, index)
        index += 1

def get_write_format(filename):
    ''' Get the write format for the image
    
    :Parameters:
    
    filename : str
               Output filename to test
    
    :Returns:
    
    out : format
            Write format for given file extension
    '''
    
    try:
        if mrc.is_writable(filename):
            return mrc
    except: pass
    for f in _formats:
        if f.is_writable(filename): return f
    return _default_write_format

def get_read_format_except(filename):
    ''' Get the write format for the image
    
    :Parameters:
    
    filename : str
               Input file to test
    
    :Returns:
    
    out : format
            Read format for given file
    '''
    
    if not os.path.exists(filename): raise IOError, "Cannot find file: %s"%(filename)
    f = get_read_format(filename)
    if f is not None: 
        #_logger.debug("Using format: %s"%str(f))
        return f
    if eman_format.is_avaliable():
        raise IOError, "Could not find format for '%s'"%filename
    else:
        raise IOError, "Could not find format for %s\n\n Installing EMAN2 adds addtional formats to Arachnid"%filename

def get_read_format(filename):
    ''' Get the write format for the image
    
    :Parameters:
    
    filename : str
               Input file to test
    
    :Returns:
    
    out : format
            Read format for given file
    '''
    
    if not os.path.exists(filename): raise IOError, "Cannot find file: %s"%(filename)
    
    for f in _formats:
        if f.is_readable(filename): return f
    return None

def cache_data():
    ''' Get keywords to be added as data cache
    
    :Returns:
    
    extra : dict
            Keyword arguments
    '''
    
    extra={}
    for f in _formats:
        if not hasattr(f, 'cache_data'): continue
        extra.update(f.cache_data())
    return extra

def readlinkabs(link):
    ''' Get the absolute path for the given symlink
    
    :Parameters:
    
    link : str
           Link filename
    
    :Returns:
    
    filename : str
               Absolute path of file link points
    '''
    
    if not os.path.exists(link): raise IOError, "Cannot find file: %s"%(link)
    if not os.path.islink(link):  return link
    p = os.readlink(link)
    if not os.path.isabs(p) and os.path.isabs(link):
        # linked file is a relative filename -> use path of shortcut
        p = os.path.join(os.path.dirname(link), p)
    if not os.path.exists(p): raise IOError, "Cannot find file: %s pointed to by link: %s"%(p, link)
    return p

def _load():
    ''' Import available formats
    '''
    
    image_formats = [mrc, spider]
    default_format=spider
    if eman_format.is_avaliable(): image_formats.append(eman_format)
    return image_formats, default_format

_formats, _default_write_format = _load()

