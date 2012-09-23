''' Read and write images in the SPIDER format

.. seealso::

    http://www.wadsworth.org/spider_doc/spider/docs/image_doc.html
    
.. todo:: define full spider header

.. todo:: support complex numbers

.. Created on Aug 9, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from arachnid.core.metadata import type_utility
import numpy

spi_defaults = dict()

header = {'nz': 1,'ny': 2,'irec':3,'iform':5,'imami':6,'fmax':7,'fmin':8, 
          'av':9, 'sig':10, 'nx':12, 'labrec':13, 'labbyt': 22, 'lenbyt':23, 
          'istack': 24, 'maxim':26, 'imgnum': 27, 'apix':38,'voltage': 39,
          'proj': 40, 'mic': 41}


spi2ara={'': ''}
#ara.update(dict([(h[0], 'mrc'+_h[0]) for h in header_image_dtype]))
ara2spi=dict([(val, key) for key,val in spi2ara.iteritems()])

spi2numpy = {
    0: numpy.uint8,
    1: numpy.int16,
    2: numpy.float32,
#    3:  complex made of two int16.  No such thing in numpy
#     however, we could manually build a complex array by reading two
#     int16 arrays somehow.
    4: numpy.complex64,

    6: numpy.uint16,    # according to UCSF
}

## mapping of numpy type to MRC mode
numpy2spi = {}

def _create_header_dtype(vtype='<f4'):
    idx2header = dict([(val, key) for key, val in header.iteritems()])
    return numpy.dtype([(idx2header.get(i, "unused_%s"%str(i+1).zfill(2)), vtype) for i in xrange(1, numpy.max(header.values())+1)])

header_dtype = _create_header_dtype()

def is_format_header(h):
    ''' Test if the given header has the proper format
    
    :Parameters:
    
    h : array
        Header to test
    
    :Returns:
    
    val : bool
          Test if dtype matches format dtype
    '''
    
    return h.dtype == header_dtype or h.dtype == header_dtype.newbyteorder()

def is_readable(filename):
    ''' Test if the file read has a valid SPIDER header
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    
    :Returns:
        
    out : bool
          True if the header conforms to SPIDER
    '''
    
    if hasattr(filename, 'dtype'): 
        h = filename
        if not is_format_header(h): 
            raise ValueError, "Array dtype incorrect"
    else: 
        try: h = read_spider_header(filename)
        except: return False
    for i in ('nz','ny','iform','nx','labrec','labbyt','lenbyt'):
        if not type_utility.is_float_int(h[i]): return False
    if not int(h['iform']) in (1,3,-11,-12,-21,-22): return False
    if (int(h['labrec'])*int(h['lenbyt'])) != int(h['labbyt']): return False
    return True

def _update_header(dest, source, header_map, tag=None):
    ''' Map values from or to the format and the internal header
    
    :Parameters:
    
    dest : array or dict
           Destination of the header values
    source : array or dict
             Source of the header values    
    header_map : dict
                 Map from destination to source
    tag : str
          Format specific attribute tag
                 
    :Returns:
    
    dest : array or dict
           Destination of the header values
    '''
    
    if source is None: return dest
    keys = dest.dtype.names if hasattr(dest, 'dtype') else dest.keys()
    tag = None
    for key in keys:
        try:
            dest[key] = source[header_map.get(key, key)]
        except:
            if tag is not None:
                try: dest[key] = source[tag+"_"+key]
                except: pass
    return dest

def _open(filename, mode):
    ''' Open a stream to filename
    
    :Parameters:
    
    filename : str
               Name of the file
    mode : str
           Mode to open file
             
    :Returns:
        
    fd : File
         File descriptor
    '''
    
    try: "+"+filename
    except: f = filename
    else:  f = open(filename, mode)
    return f

def _close(filename, fd):
    ''' Close the file descriptor (if it was opened by caller)
    
    filename : str
               Name of the file
    fd : File
         File descriptor
    '''
    
    if fd != filename: fd.close()
    
def read_header(filename, index=None):
    ''' Read the SPIDER header
    
    .. todo:: remove and move to read_spider_header (follow mrc)
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    index : int, optional
            Index of image to get the header, if None, the stack header (Default: None)
    
    :Returns:
        
    header : dict
             Dictionary with header information
    '''
    
    h = read_spider_header(filename, index)
    header={}
    header['apix'] = h['apix']
    return header

def read_spider_header(filename, index=None):
    ''' Read the SPIDER header
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    index : int, optional
            Index of image to get the header, if None, the stack header (Default: None)
    
    :Returns:
        
    out : array
          Array with header information in the file
    '''
    
    f = _open(filename, 'r')
    try:
        curr = f.tell()
        h = numpy.fromfile(f, dtype=header_dtype, count=1)
        if not is_readable(h):
            f.seek(curr)
            h = numpy.fromfile(f, dtype=header_dtype.newbyteorder(), count=1)
        if not is_readable(h): raise IOError, "Not a SPIDER file"
        if index is not None:
            h_len = int(h['labbyt'])
            i_len = int(h['nx']) * int(h['ny']) * int(h['nz']) * 4
            count = max(int(h['istack']), 1)
            if index >= count: raise IOError, "Index exceeds number of images in stack: %d < %d"%(index, count)
            offset = index * (h_len+i_len)
            f.seek(offset)
            h = numpy.fromfile(f, dtype=h.dtype, count=1)
    finally:
        _close(filename, f)
    return h

def read_image(filename, index=None, header=None):
    ''' Read an image from the specified file in the SPIDER format
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    index : int, optional
            Index of image to get, if None, first image (Default: None)
    header : dict, optional
             Dictionary to hold header values
             
    :Returns:
        
    out : array
          Array with image information from the file
    '''
    
    f = _open(filename, 'r')
    h = None
    try:
        if index is None: index = 0
        h = read_spider_header(f)
        if header is not None: _update_header(header, h, spi2ara, 'mrc')
        h_len = int(h['labbyt'])
        d_len = int(h['nx']) * int(h['ny']) * int(h['nz'])
        i_len = d_len * 4
        count = count_images(h)
        if index >= count: raise IOError, "Index exceeds number of images in stack: %d < %d"%(index, count)
        offset = h_len + index * (h_len+i_len)
        f.seek(offset)
        out = numpy.fromfile(f, dtype=h.dtype.fields['nx'][0], count=d_len)
        if int(h['nz']) > 1:   out = out.reshape(int(h['nx']), int(h['ny']), int(h['nz']))
        elif int(h['ny']) > 1: out = out.reshape(int(h['nx']), int(h['ny']))
    finally:
        _close(filename, f)
    return out

def iter_images(filename, index=None, header=None):
    ''' Read a set of SPIDER images
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    index : int, optional
            Index of image to start, if None, start with the first image (Default: None)
    header : dict, optional
             Dictionary to hold header values
    
    :Returns:
        
    out : array
          Array with image information from the file
    '''
    
    f = _open(filename, 'r')
    if index is None: index = 0
    try:
        h = read_spider_header(f)
        if header is not None: _update_header(header, h, spi2ara, 'mrc')
        h_len = int(h['labbyt'])
        d_len = int(h['nx']) * int(h['ny']) * int(h['nz'])
        i_len = d_len * 4
        count = count_images(h)
        if numpy.any(index >= count): raise IOError, "Index exceeds number of images in stack: %d < %d"%(index, count)
        offset = h_len + index * (h_len+i_len)
        f.seek(offset)
        if not hasattr(index, '__iter__'): index =  xrange(index, count)
        else: index = index.astype(numpy.int)
        for i in index:
            out = numpy.fromfile(f, dtype=h.dtype.fields['nx'][0], count=d_len)
            if int(h['nz']) > 1:   out = out.reshape(int(h['nx']), int(h['ny']), int(h['nz']))
            elif int(h['ny']) > 1: out = out.reshape(int(h['nx']), int(h['ny']))
            yield out
    finally:
        _close(filename, f)

def count_images(filename):
    ''' Count the number of images in the file
    
    :Parameters:
    
    filename : str or file objectfou
               Filename or open stream for a file
    
    :Returns:
        
    out : int
          Number of images in the file
    '''
    
    if hasattr(filename, 'dtype'): h=filename
    else: h = read_spider_header(filename)
    return max(int(h['istack']), 1)

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
    
    return False

def write_image(filename, img, index=None, header=None):
    ''' Write an image array to a file in the MRC format
    
    :Parameters:
    
    filename : str
               Name of the output file
    img : array
          Image array
    index : int, optional
            Index to write image in the stack
    header : dict, optional
             Dictionary of header values
    '''
    
    
    try: img = img.type(spi2numpy[numpy2spi[img.dtype.type]])
    except: raise TypeError, "Unsupported type for SPIDER writing: %s"%str(img.dtype)
    
    f = _open(filename, 'w')
    try:
        if header is None or not hasattr(header, 'dtype') or not is_format_header(header):
            h = numpy.zeros(1, header_dtype)
            _update_header(h, spi_defaults, ara2spi)
            header=_update_header(h, header, ara2spi, 'spi')
        if index is not None:
            header.tofile(f)
            # convert to image header
            #_update_header(h, spi_defaults, ara2spi)
            #header=_update_header(h, header, ara2spi, 'spi')
    finally:
        _close(filename, f)




