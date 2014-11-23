''' Arachnid Image format

@todo - transpose projection in volume

This format describes a binary data that comprises a required fixed header (79 bytes) followed by an optional 
variable-length header and then by the image data.

Header
------

Header Length: 78 bytes

Name             Type       Size (Bytes)    Description
====             ====       ============    ===========
magic            String     9               Identifies the format [Required to be 'WEBFORMAT']
nx               Integer    8               Size of image/volume in the x-direction (width)
ny               Integer    8               Size of image/volume in the y-direction (height)
nz               Integer    8               Size of image/volume in the z-direction (depth)
count            Integer    8               Number of images/volumes
dtype            String     2               Two character string describing the data type (see below for description)
byte_num         Integer    2               Integer size of each pixel in bytes
order            String     1               Single character specifying row-major (c-contiguous) or column-major (fortran-contiguous)
pixelSpacing     Float      4               Sampling rate (pixel/Angstroms) [0 if not specified]
extended_ident   String     20              Identifies the extended header [Optional]
extended         Integer    8               Number of bytes in the extended header [0 if not extended header]

The dtype field in the header describes a two character code, e.g. <f.

    - The first character describes the byte order or endian of the data
      Character  Description
      =========  ===========
      '<'        little-endian
      '>'        big-endian
      '|'        not applicable

    - The second character describes the data type
      Character  Description
      =========  ===========
      'b'        Boolean
      'i'        (signed) integer
      'u'        unsigned integer
      'f'        floating-point
      'c'        complex-floating point
      'S',       string
      'U'        unicode
      'V'        raw data (void)

The `order` field in the header describes the layout of a 2 or 3D array in memory. Both
c-contiguous or row-major and fortran-contiguous or column major are supported.

Extended Image Header
---------------------

Name             Type       Size (Bytes)    Description
====             ====       ============    ===========
defocus          Float           8          Defocus of the image
astig_ang        Float           8          Astigmatism Angle (SPIDER convention)
astig_mag        Float           8          Astigmatism Magnitude (SPIDER convention)
type             Char            4          Type of image

Allow image `type`s are:

    - Power Spectra: 'P'
    - Windowed Particle: 'W'
    - Micrograph: 'M'
 
 This follows the standard of NumPy:
 http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#arrays-dtypes

.. Created on Jul 18, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy, logging, os
import util

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

header_dtype = numpy.dtype([
 ('magic', 'S10'),
 ('nx', numpy.int64),
 ('ny', numpy.int64),
 ('nz', numpy.int64),
 ('count', numpy.int64),
 ('dtype', 'S3'),
 ('byte_num', numpy.int16),
 ('order', 'S2'),
 ('pixelSpacing', numpy.float32),
 ('extended_ident', 'S20'),
 ('extended', numpy.int64),]
)

metadata_dtype = numpy.dtype([
 ('amplitudeContrast', numpy.float64),
 ('cs', numpy.float64),
 ('defocusU', numpy.float64),
 ('defocusV', numpy.float64),
 ('defocusUAngle', numpy.float64),
 ('voltage', numpy.float64),
 ('micrograph', numpy.int64),
 ('xcoord', numpy.int64),
 ('ycoord', numpy.int64),
 ('xshift', numpy.float64),
 ('yshift', numpy.float64),
 ('qx', numpy.float64),
 ('qy', numpy.float64),
 ('qz', numpy.float64),
 ('qw', numpy.float64),]
)

def create_header(shape, dtype, order='C', header=None):
    ''' Create a header for the web image format
    
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
    
    h = numpy.zeros(1, header_dtype)
    h['magic']='WEBFORMAT'
    h['nx']=shape[0]
    h['ny']=shape[1] if len(shape) > 1 else 1
    h['nz']=shape[2] if len(shape) > 2 else 1
    h['count']=shape[3] if len(shape) > 3 else 1 
    h['dtype']= dtype.str[:2]
    h['byte_num']=int(dtype.str[2:])
    h['order']=order
    h['pixelSpacing']=header.get('pixelSpacing', 0.0) if header is not None else 0.0
    h['extended_ident']=""
    h['extended']=0
    return h

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
    
    imheader = dict(pixelSpacing=header['pixelSpacing'][0])
    
    dtype = numpy.dtype(header['dtype'][0]+str(header['byte_num'][0]))
    shape = (header['nx'][0], header['ny'][0], header['nz'][0])
    if header_dtype.newbyteorder()[0]==header.dtype[0]: dtype = dtype.newbyteorder()
    return (int(header_dtype.itemsize+int(header['extended'])),
           (imheader, dtype, numpy.prod(shape), shape, 
           header_dtype.newbyteorder()[0]==header.dtype[0], header['order'][0],) )

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
    
    return h.dtype == header_dtype

def read_web_header(filename, index=None):
    ''' Read the WEB header
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    index : int, ignored
            Index of image to get the header, if None, the stack header (Default: None)
    
    :Returns:
        
    out : array
          Array with header information in the file
    '''
    
    f = util.uopen(filename, 'rb')
    m=None
    try:
        #curr = f.tell()
        h = numpy.fromfile(f, dtype=header_dtype, count=1)
        if not is_readable(h): h = h.byteswap().newbyteorder()
        if not is_readable(h): raise IOError, "Not an WEB file"
        if h['extended_ident'] == 'WEBMETADATA':
            count = h['extended'][0]/metadata_dtype.itemsize
            if (count*metadata_dtype.itemsize) != h['extended'][0]:
                _logger.warn("Unable to read metadata - size mismatch: %d *%d = %d != %d"%(count, metadata_dtype.itemsize, (count*metadata_dtype.itemsize), h['extended'][0]))
            else:
                m = numpy.fromfile(f, dtype=metadata_dtype, count=count)
    finally:
        util.close(filename, f)
    return h, m

def is_readable(filename):
    ''' Test if the file read has a valid WEB header
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    
    :Returns:
        
    out : bool
          True if the header conforms to WEB
    '''
    
    if hasattr(filename, 'dtype') or (isinstance(filename, tuple) and hasattr(filename[0], 'dtype')): 
        h = filename[0] if isinstance(filename, tuple) else filename
        if not is_format_header(h): raise ValueError, "Array dtype incorrect"
    else: 
        try: h = read_web_header(filename)[0]
        except: return False
    
    if h['magic'] != 'WEBFORMAT': return False
    if not numpy.alltrue([h[v][0] > 0 for v in ('nx', 'ny', 'nz', 'count')]): return False
    if h['dtype'][0] not in ('<', '>', '|'): return False
    if h['dtype'][1] not in ('t', 'b', 'i', 'u', 'f', 'c', 'o', 'S', 'U', 'V'): return False
    if h['order'][0] not in ('C', 'F'): return False
    if not (h['byte_num'][0] > 0): return False
    return True

def count_images(filename):
    ''' Count the number of images in the file
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    
    :Returns:
        
    out : int
          Number of images in the file
    '''
    
    if hasattr(filename, 'dtype'): h=filename
    else: h = read_web_header(filename)
    return h['count'][0]

def read_header(filename, index=None):
    ''' Read the WEB header
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    index : int, ignored
            Index of image to get the header, if None, the stack header (Default: None)
    
    :Returns:
        
    header : dict
             Dictionary with header information
    '''
    
    h = read_web_header(filename, index)
    header={}
    header['apix']=float(h['pixelSpacing'][0])
    header['count'] = int(h['nz'][0]) if int(h['nz'][0])!=int(h['nx'][0]) else 1
    header['nx'] = int(h['nx'][0])
    header['ny'] = int(h['ny'][0])
    header['nz'] = int(h['nz'][0]) if int(h['nz'][0])==int(h['nx'][0]) else 1
    for key in h.dtype.fields.iterkeys():
        header['web_'+key] = h[key][0]
    header['format'] = 'mrc'
    return header

def valid_image(filename):
    ''' Test if the image is valid
    
    :Parameters:
    
        filename : str
                   Input filename to test
    
    :Returns:
        
        flag : bool
               True if image is valid
    '''
    
    f = util.uopen(filename, 'rb')
    try:
        h = read_web_header(f)
        offset, ar_args = array_from_header(h)
        return file_size(f) == (offset + h['count'] * ar_args[1] * ar_args[0].itemsize)
    finally:
        util.close(filename, f)

def file_size(fileobject):
    fileobject.seek(0,2) # move the cursor to the end of the file
    size = fileobject.tell()
    return size

def read_image(filename, index=None, header=None, cache=None):
    ''' Read an image from the specified file in the WEB format
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    index : int, optional
            Index of image to get, if None, first image (Default: None)
    header : dict, optional
             Output dictionary to place header values
    
    :Returns:
        
    out : array
          Array with image information from the file
    '''
    
    idx = 0 if index is None else index
    f = util.uopen(filename, 'rb')
    try:
        h = read_web_header(f)
        #if header is not None: util.update_header(header, h, web2ara, 'web')
        if idx >= count_images(h): raise IOError, "Index exceeds number of images in stack: %d < %d"%(idx, count_images(h))
        offset, ar_args = array_from_header(h)
        f.seek(offset + idx * ar_args[1] * ar_args[0].itemsize)
        out = util.read_image(f, *ar_args)
    finally:
        util.close(filename, f)
    return out

def iter_images(filename, index=None, header=None):
    ''' Read a set of SPIDER images
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    index : int, optional
            Index of image to start, if None, start with the first image (Default: None)
    header : dict, optional
             Output dictionary to place header values
    
    :Returns:
        
    out : array
          Array with image information from the file
    '''
    
    f = util.uopen(filename, 'rb')
    if index is None: index = 0
    try:
        h = read_web_header(f)
        #if header is not None: util.update_header(header, h, web2ara, 'web')
        count = count_images(h)
        offset, ar_args = array_from_header(h)
        f.seek(int(offset))
        if not hasattr(index, '__iter__'): index =  xrange(index, count)
        else: index = index.astype(numpy.int)
        for i in index:
            yield util.read_image(f, *ar_args)
    finally:
        util.close(filename, f)

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
    return ext == 'web'

def write_image(filename, img, index=None, header=None, inplace=False):
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
    inplace : bool
              Write new image to stack without removing the stack
    '''
    if header is None and hasattr(img, 'header'): header=img.header
    
    mode = 'rb+' if index is not None and (index > 0 or inplace and index > -1) else 'wb+'
    f = util.uopen(filename, mode)
    if header is None or not is_format_header(header):
        header = create_header(img.shape, img.dtype, img.order, header)
    try:
        if inplace:
            f.seek(int(header.itemsize+int(header['extended'])+index*img.ravel().shape[0]*img.dtype.itemsize))
        elif f != filename:
            f.seek(0)
            header.tofile(f)
            if index > 0: f.seek(int(header.itemsize+int(header['extended'])+index*img.ravel().shape[0]*img.dtype.itemsize))
        img.tofile(f)
    finally:
        util.close(filename, f)


