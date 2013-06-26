''' Reading and writing NumPY arrays in various image formats

Supported formats:
    
     - :py:mod:`EMAN2/SPARX <formats.eman_format>`
     - :py:mod:`MRC <formats.mrc>`
     - :py:mod:`SPIDER <formats.spider>`
     
.. todo:: mrc format gives nans with direct detector data, disabled until fixed

.. Created on Aug 11, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import logging, os
from ..app import tracing
from formats import spider, eman_format as spider_writer, mrc, ccp4
from ..metadata import spider_utility, format_utility
from ..parallel import process_tasks, process_queue, mpi_utility
import scipy.io
import numpy
mrc;
ccp4;

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
    img1 = read_image(filename, index)
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
    for row, data in process_tasks.for_process_mp(iter_images(filename, label), image_processor, img1.shape, queue_limit=100, **extra):
        mat[row, :] = data.ravel()[:img.shape[0]]
    if force_mat:
        scipy.io.savemat(cache_file, dict(data=mat, label=label), oned_as='column', format='5')
    else:
        if cache_file is not None and cache_file != "":
            _logger.info("Caching image matrix")
            scipy.io.savemat(cache_file, dict(coo=numpy.asarray(mat.shape, dtype=numpy.int), dtype=mat.dtype.name), oned_as='column', format='5')
            mat.ravel().tofile(cache_dat)
    return shmem_mat

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
    
    if is_spider_format(filename) and os.path.splitext(filename)[1] == os.path.splitext(tempfile)[1]: return filename
    
    img = read_image(filename, index)
    spider_writer.write_spider_image(tempfile, img)
    #for index, img in enumerate(iter_images(filename)):
    #    spider_writer.write_image(tempfile, img, index)
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
    
    if not os.path.exists(filename): raise IOError, "Cannot find file: %s"%filename
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
    
    format = get_read_format_except(filename)
    return format.read_header(filename, index)

def read_image(filename, index=None, **extra):
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
    
    try:
        filename = readlinkabs(filename)
    except:
        _logger.error("Problem with: %s"%str(filename))
        raise
    format = get_read_format_except(filename)
    return format.read_image(filename, index, **extra)

def readlinkabs(link):
    ''' Get the absolute path for the given symlink
    
    :Parameters:
    
    link : str
           Link filename
    
    :Returns:
    
    filename : str
               Absolute path of file link points
    '''
    
    if not os.path.islink(link):  return link
    p = os.readlink(link)
    if os.path.isabs(p): return p
    return os.path.join(os.path.dirname(link), p)

def process_images(input_file, output_file, transform_func, index=None, **extra):
    ''' Apply functor to each image in the list
    '''
    
    for i, img in enumerate(iter_images(input_file, index)):
        img = transform_func((i, img), **extra)
        write_image(output_file, img, i)

def iter_images(filename, index=None):
    ''' Read a set of images from the given file
    
    :Parameters:
    
    filename : str
               Input filename to read
    index : int or array, optional
            Index of image to start or array of selected images, if None, start with the first image (Default: None)
    
    :Returns:
        
    out : array
          Array with header information in the file
        
    .. todo:: iter single images
    '''
    
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
                    filename = spider_utility.spider_filename(filename, int(index[i, 0]))
                else:
                    filename = spider_utility.spider_filename(filename, int(index[i]))
                yield read_image(filename)
            return
        elif index.ndim == 2 and index.shape[1]>1:
            beg = 0
            tot = len(numpy.unique(index[:, 0].astype(numpy.int)))
            if not isinstance(filename, dict) and not hasattr(filename, 'find'): filename=filename[0]
            for i in xrange(tot):
                id = index[beg, 0]
                filename = spider_utility.spider_filename(filename, int(id)) if not isinstance(filename, dict) else filename[int(id)]
                sel = numpy.argwhere(id == index[:, 0]).ravel()
                if beg != sel[0]: raise ValueError, "Array must be sorted by file ids: %d != %d -- %f, %f"%((beg), sel[0], index[beg, 0], beg)
                try:
                    for img in iter_images(filename, index[sel, 1]):
                        yield img
                except:
                    _logger.error("stack filename: %s - %d to %d"%(filename, numpy.min(index[sel, 1]), numpy.max(index[sel, 1])))
                    raise
                beg += sel.shape[0]
            
            '''
            fileid = index[:, 0].astype(numpy.int)
            ids = numpy.unique(fileid)
            if not isinstance(filename, dict) and not hasattr(filename, 'find'): filename=filename[0]
            for id in ids:
                filename = spider_utility.spider_filename(filename, int(id)) if not isinstance(filename, dict) else filename[int(id)]
                for img in iter_images(filename, index[id == fileid, 1]):
                    yield img
            '''
            return
        
        
    if index is not None and hasattr(index, '__iter__') and not hasattr(index, 'ndim'): index = numpy.asarray(index)
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
    
        format = get_read_format_except(filename[0])
        total = 0
        for f in filename:
            total += format.count_images(f)
        return total
    else:
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

def write_image(filename, img, index=None, header=None):
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
    '''
    
    if index is not None and index == 0 and os.path.exists(filename):
        os.unlink(filename)
    
    format = get_write_format(filename)
    if format is None: 
        raise IOError, "Could not find format for extension of %s"%filename
    format.write_image(filename, img, index, header)
    
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
    
    write : format
            Write format for given file extension
    '''
    
    for f in _formats:
        if f.is_writable(filename): return f
    return _default_write_format

def get_read_format_except(filename):
    ''' Get the write format for the image
    
    :Parameters:
    
    filename : str
               Input file to test
    
    :Returns:
    
    write : format
            Read format for given file
    '''
    
    if not os.path.exists(filename): raise IOError, "Cannot find file: %s"%filename
    f = get_read_format(filename)
    if f is not None: 
        #_logger.debug("Using format: %s"%str(f))
        return f
    raise IOError, "Could not find format for %s"%filename

def get_read_format(filename):
    ''' Get the write format for the image
    
    :Parameters:
    
    filename : str
               Input file to test
    
    :Returns:
    
    write : format
            Read format for given file
    '''
    
    try:
        if mrc.is_readable(filename) and mrc.count_images(filename) > 1 and not mrc.is_volume(filename):
            return mrc
    except: pass
    for f in _formats:
        if f.is_readable(filename): return f
    return None

def _load():
    ''' Import available formats
    '''
    
    #from formats import mrc
    formats = []#mrc]
    try: from formats import eman_format
    except: tracing.log_import_error("Cannot load EMAN2 - supported image formats will not be available - see documentation for more details")
    else: formats.append(eman_format)
    if len(formats) == 0: raise ImportError, "No image format modules loaded!"
    return formats, eman_format

_formats, _default_write_format = _load()

