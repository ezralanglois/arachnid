''' Read and write images in the SPIDER format

.. seealso::

    http://www.wadsworth.org/spider_doc/spider/docs/image_doc.html
    
.. todo:: define full spider header

.. Created on Aug 9, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from arachnid.core.metadata import type_utility
import numpy, os, logging
import util

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

spi_defaults = dict()

_header_map = {'nz': 1,'ny': 2,'irec':3,'iform':5,'imami':6,'fmax':7,'fmin':8, 
          'av':9, 'sig':10, 'nx':12, 'labrec':13, 'labbyt': 22, 'lenbyt':23, 
          'istack': 24, 'maxim':26, 'imgnum': 27, 'apix':38,'voltage': 39,
          'proj': 40, 'mic': 41}

spi2ara={'apix': 'apix'}
#ara.update(dict([(h[0], 'mrc'+_h[0]) for h in header_image_dtype]))
ara2spi=dict([(val, key) for key,val in spi2ara.iteritems()])

spi2numpy = {
     1: numpy.float32,
     3: numpy.float32,
    -11: numpy.complex64,
    -12: numpy.complex64, #Even
    -21: numpy.complex64,
    -22: numpy.complex64, #Even
}
'''

spi2numpy = {
     1: numpy.float64,
     3: numpy.float64,
    -11: numpy.complex128,
    -12: numpy.complex128, #Even
    -21: numpy.complex128,
    -22: numpy.complex128, #Even
}
'''

## mapping of numpy type to MRC mode
numpy2spi = {
}

def _create_header_dtype(vtype='<f4'):
    idx2header = dict([(val, key) for key, val in _header_map.iteritems()])
    return numpy.dtype([(idx2header.get(i, "unused_%s"%str(i+1).zfill(2)), vtype) for i in xrange(1, numpy.max(_header_map.values())+1)])

header_dtype = _create_header_dtype()


def create_header(shape, dtype, order='C', header=None):
    ''' Create a header for the SPIDER image format
    
    @todo support header parameters
    
    :Parameters:
    
    shape : tuple
            Shape of the array 
    dtype : numpy.dtype 
            Data type for NumPy ndarray
    header : dict
             Header values  for image
    :Returns:
    
    h : dtype
        Data type for NumPy ndarray describing the header
    '''
    
    pass

def array_from_header(header):
    ''' Convert header information to array parameters
    
    :Parameters:
    
    header : header_dtype
             Header fields
    
    :Returns:
    
    header : dict
             File header
    dtype : dtype
            Data type
    shape : tuple
            Shape of the array
    order : str
            Order of the array
    offset : int
             Header offset
    swap : bool
            Swap byte order
    '''
    
    pass

def cache_data():
    ''' Get keywords to be added as data cache
    
    :Returns:
    
    extra : dict
            Keyword arguments
    '''
    
    return dict()

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
        if not type_utility.is_float_int(h[i]):  
            return False
    if not int(h['iform']) in (1,3,-11,-12,-21,-22): 
        return False
    if (int(h['labrec'])*int(h['lenbyt'])) != int(h['labbyt']): 
        return False
    return True
    
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
    header['apix'] = float(h['apix'][0])
    header['fourier_even'] = (h['iform'][0] == -12 or h['iform'][0] == -22)
    
    
    header['count'] = max(int(h['maxim'][0]), int(h['istack'][0]))
    header['nx'] = int(h['nx'][0])
    header['ny'] = int(h['ny'][0])
    header['nz'] = int(h['nz'][0])
    for key in h.dtype.fields.iterkeys():
        header['spi_'+key] = float(h[key][0])
    header['format'] = 'SPIDER'
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
    
    f = util.uopen(filename, 'r')
    try:
        #curr = f.tell()
        h = numpy.fromfile(f, dtype=header_dtype, count=1)
        if not is_readable(h):
            h = h.newbyteorder()
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
        util.close(filename, f)
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
    
    f = util.uopen(filename, 'r')
    h = None
    try:
        if index is None: index = 0
        h = read_spider_header(f)
        dtype = numpy.dtype(spi2numpy[float(h['iform'])])
        try:
            if header_dtype.newbyteorder()==h.dtype: dtype = dtype.newbyteorder()
        except:
            _logger.error("dtype: %s"%str(dtype))
            raise
        #if header is not None: util.update_header(header, h, spi2ara, 'spi')
        if header is not None: header.update(read_header(h))
        
        h_len = int(h['labbyt'])
        d_len = int(h['nx']) * int(h['ny']) * int(h['nz'])
        i_len = d_len * 4
        
        count = count_images(h)
        
        if index >= count: raise IOError, "Index exceeds number of images in stack: %d < %d"%(index, count)
        if int(h['istack']) > 0:
            offset = h_len*2 + index * (h_len+i_len)
        else:
            if count > 1: raise ValueError, "Improperly formatted SPIDER header - not stack but contains mutliple images"
            offset = h_len
        if count > 1:
            if file_size(f) != (h_len + count * (h_len+i_len)): raise ValueError, "file size != header: %d != %d - %d"%(file_size(f), (h_len + count * (h_len+i_len)), count)
        else:
            if file_size(f) != (h_len + count * i_len): raise ValueError, "file size != header: %d != %d - %d"%(file_size(f), (h_len + count * (h_len+i_len)), count)
        f.seek(offset)
        out = numpy.fromfile(f, dtype=dtype, count=d_len)
        #assert(out.ravel().shape[0]==d_len)
        if int(h['nz']) > 1:   out = out.reshape(int(h['nz']), int(h['ny']), int(h['nx']))
        elif int(h['ny']) > 1: 
            try:
                out = out.reshape(int(h['ny']), int(h['nx']))
            except:
                _logger.error("%d != %d*%d = %d"%(out.ravel().shape[0], int(h['nx']), int(h['ny']), int(h['nx'])*int(h['ny'])))
                raise
    finally:
        util.close(filename, f)
    #if header_image_dtype.newbyteorder()==h.dtype:out = out.byteswap()
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
    
    f = util.uopen(filename, 'r')
    if index is None: index = 0
    try:
        h = read_spider_header(f)
        dtype = numpy.dtype(spi2numpy[float(h['iform'])])
        if header_dtype.newbyteorder()==h.dtype: dtype = dtype.newbyteorder()
        if header is not None: util.update_header(header, h, spi2ara, 'spi')
        h_len = int(h['labbyt'])
        d_len = int(h['nx']) * int(h['ny']) * int(h['nz'])
        i_len = d_len * 4
        count = count_images(h)
        if numpy.any(index >= count):  raise IOError, "Index exceeds number of images in stack: %s < %d"%(str(index), count)
        offset = h_len + 0 * (h_len+i_len)
        f.seek(offset)
        if not hasattr(index, '__iter__'): index =  xrange(index, count)
        else: index = index.astype(numpy.int)
        last=0
        for i in index:
            if i != (last+1): f.seek(int(h_len + i * (h_len+i_len)))
            out = numpy.fromfile(f, dtype=dtype, count=d_len)
            if int(h['nz']) > 1:   out = out.reshape(int(h['nz']), int(h['ny']), int(h['nx']))
            elif int(h['ny']) > 1: out = out.reshape(int(h['ny']), int(h['nx']))
            yield out
    finally:
        util.close(filename, f)

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
    return max(int(h['maxim']), 1)

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
    
    ext = os.path.splitext(filename)[1][1:].lower()
    return ext == 'spi'

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
    
    #float64
    #complex64
    
    if header is None and hasattr(img, 'header'): header=img.header
    dtype = numpy.complex64 if numpy.iscomplexobj(img) else numpy.float32
    try: img = img.astype(dtype)
    except: raise TypeError, "Unsupported type for SPIDER writing: %s"%str(img.dtype)
    
    mode = 'a' if index is not None and index > 0 else 'w'
    f = util.uopen(filename, mode)
    try:
        if header is None or not hasattr(header, 'dtype') or not is_format_header(header):
            h = numpy.zeros(1, header_dtype)
            even = header['fourier_even'] if header is not None and 'fourier_even' in header else None
            util.update_header(h, spi_defaults, ara2spi)
            header=util.update_header(h, header, ara2spi, 'spi')
            
            # Image size in header
            header['nx'] = img.T.shape[0]
            header['ny'] = img.T.shape[1] if img.ndim > 1 else 1
            header['nz'] = img.T.shape[2] if img.ndim > 2 else 1
            
            header['lenbyt'] = img.shape[0]*4
            header['labrec'] = 1024 / int(header['lenbyt'])
            if 1024%int(header['lenbyt']) != 0: 
                header['labrec'] = int(header['labrec'])+1
            header['labbyt'] = int(header['labrec'] ) * int(header['lenbyt'])
            
            # 
            #header['irec']
            if numpy.iscomplexobj(img):
                header['iform'] = 3 if img.ndim == 3 else 1
                # determine even or odd Fourier - assumes other dim are padded appropriately
                if even is None:
                    v = int(round(float(img.shape[1])/img.shape[0]))
                    v = img.shape[1]/v
                    even = (v%2)==0
                if even:
                    header['iform'] = -22  if img.ndim == 3 else -12 
                else:
                    header['iform'] = -21  if img.ndim == 3 else -11 
            else:
                header['iform'] = 3 if img.ndim == 3 else 1 
        
        fheader = numpy.zeros(header['labbyt']/4, dtype=numpy.float32)
        for name, idx in _header_map.iteritems(): 
            fheader[idx-1]=float(header[name])
        
        if index is not None:
            fheader[_header_map['maxim']] = index+1
            fheader[_header_map['imgnum']] = index+1
            fheader[_header_map['istack']] = 2
            f.seek(0)
            fheader.tofile(f)
            fheader[_header_map['istack']] = 0
            f.seek(index * (int(header['nx']) * int(header['ny']) * int(header['nz']) * 4 + int(header['labbyt'])))
        fheader.tofile(f)
        img.tofile(f)
    finally:
        util.close(filename, f)


def file_size(fileobject):
    fileobject.seek(0,2) # move the cursor to the end of the file
    size = fileobject.tell()
    return size



