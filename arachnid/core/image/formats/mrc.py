''' Read and write images in the MRC format

.. todo:: define arachnid header and map to mrc

.. todo: finish reading

.. todo:: finish tests


.. Created on Aug 9, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy, sys
from spider import _open, _close, _update_header

mrc2numpy = {
    0: numpy.uint8,
    1: numpy.int16,
    2: numpy.float32,
    3: numpy.dtype([('real', numpy.int16), ('imag', numpy.int16)]), 
      #complex made of two int16.  No such thing in numpy
#     however, we could manually build a complex array by reading two
#     int16 arrays somehow.
    4: numpy.complex64,

    6: numpy.uint16,    # according to UCSF
}

## mapping of numpy type to MRC mode
numpy2mrc = {
    ## convert these to int8
    numpy.uint8: 0,
    numpy.bool: 0,
    numpy.bool_: 0,

    ## convert these to int16
    numpy.int16: 1,
    numpy.int8: 1,

    ## convert these to float32
    numpy.float32: 2,
    numpy.float64: 2,
    numpy.int32: 2,
    numpy.int64: 2,
    numpy.int: 2,
    numpy.uint32: 2,
    numpy.uint64: 2,

    ## convert these to complex64
    numpy.complex: 4,
    numpy.complex64: 4,
    numpy.complex128: 4,

    ## convert these to uint16
    numpy.uint16: 6,
}
'''
    if(IsLittleEndian())
    {
        machine_stamp[0] = 68;
        machine_stamp[1] = 65;
    }
    else
    {
        machine_stamp[0] = machine_stamp[1] = 17;
    }
'''
intbyteorder = {
    0x11110000: 'big',
    0x44440000: 'little', #hack
    0x44410000: 'little'
}
byteorderint = {
    'big': 0x11110000,
    'little': 0x44410000
}

mrc_defaults = dict(alpha=90, beta=90, gamma=90, mapc=1, mapr=2, maps=3, map='MAP ', byteorder=byteorderint[sys.byteorder])

def _gen_header():
    ''' Create the header for an MRC image and stack
    
    .. note::
        
        The following code was adopted from mrc.py in Scripps Appion Package
    
    :Returns:
    
    header_image_dtype : numpy.dtype
                         Header for an MRC image
    header_stack_dtype : numpy.dtype
                         Header for an MRC stack
    '''
    
    shared_fields = [
        ('nx', numpy.int32),
        ('ny', numpy.int32),
        ('nz', numpy.int32),
        ('mode', numpy.int32),
        ('nxstart', numpy.int32),
        ('nystart', numpy.int32),
        ('nzstart', numpy.int32),
        ('mx', numpy.int32),
        ('my', numpy.int32),
        ('mz', numpy.int32),
        ('xlen', numpy.float32),
        ('ylen', numpy.float32),
        ('zlen', numpy.float32),
        ('alpha', numpy.float32),
        ('beta', numpy.float32),
        ('gamma', numpy.float32),
        ('mapc', numpy.int32),
        ('mapr', numpy.int32),
        ('maps', numpy.int32),
        ('amin', numpy.float32),
        ('amax', numpy.float32),
        ('amean', numpy.float32),
        ('ispg', numpy.int32),
        ('nsymbt', numpy.int32),
    ]
    
    header_image_dtype = numpy.dtype( shared_fields+[
        ('extra', 'S100'),
        ('xorigin', numpy.float32),
        ('yorigin', numpy.float32),
        ('zorigin', numpy.float32),
        ('map', 'S4'),
        ('byteorder', numpy.int32),
        ('rms', numpy.float32),
        ('nlabels', numpy.int32),
        ('label0', 'S80'),
        ('label1', 'S80'),
        ('label2', 'S80'),
        ('label3', 'S80'),
        ('label4', 'S80'),
        ('label5', 'S80'),
        ('label6', 'S80'),
        ('label7', 'S80'),
        ('label8', 'S80'),
        ('label9', 'S80'),
    ])
    
    header_stack_dtype = numpy.dtype( shared_fields+[
        ("dvid", numpy.uint16),
        ("nblank", numpy.uint16),
        ("itst", numpy.int32),
        ("blank", 'S24'),
        ("nintegers", numpy.uint16),
        ("nfloats", numpy.uint16),
        ("sub", numpy.uint16),
        ("zfac", numpy.uint16),
        ("min2", numpy.float32),
        ("max2", numpy.float32),
        ("min3", numpy.float32),
        ("max3", numpy.float32),
        ("min4", numpy.float32),
        ("max4", numpy.float32),
        ("type", numpy.uint16),
        ("lensum", numpy.uint16),
        ("nd1", numpy.uint16),
        ("nd2", numpy.uint16),
        ("vd1", numpy.uint16),
        ("vd2", numpy.uint16),
        ("min5", numpy.float32),
        ("max5", numpy.float32),
        ("numtimes", numpy.uint16),
        ("imgseq", numpy.uint16),
        ("xtilt", numpy.float32),
        ("ytilt", numpy.float32),
        ("ztilt", numpy.float32),
        ("numwaves", numpy.uint16),
        ("wave1", numpy.uint16),
        ("wave2", numpy.uint16),
        ("wave3", numpy.uint16),
        ("wave4", numpy.uint16),
        ("wave5", numpy.uint16),
        ("xorigin", numpy.float32),
        ("yorigin", numpy.float32),
        ("zorigin", numpy.float32),
        ("nlabels", numpy.int32),
        ('label0', 'S80'),
        ('label1', 'S80'),
        ('label2', 'S80'),
        ('label3', 'S80'),
        ('label4', 'S80'),
        ('label5', 'S80'),
        ('label6', 'S80'),
        ('label7', 'S80'),
        ('label8', 'S80'),
        ('label9', 'S80'),
    ] )
    
    return header_image_dtype, header_stack_dtype

# --------------------------------------------------------------------
# End attribution
# --------------------------------------------------------------------

header_image_dtype, header_stack_dtype = _gen_header()

mrc2ara={'': ''}
mrc2ara.update(dict([(h[0], 'mrc'+h[0]) for h in header_image_dtype.names]))
mrc2ara.update(dict([(h[0], 'mrc'+h[0]) for h in header_stack_dtype.names]))
ara2mrc=dict([(val, key) for key,val in mrc2ara.iteritems()])

def is_format_header(h):
    ''' Test if the given header has the proper format
    
    :Parameters:
    
    h : array
        Header to test
    
    :Returns:
    
    val : bool
          Test if dtype matches format dtype
    '''
    
    return h.dtype == header_stack_dtype or h.dtype == header_stack_dtype.newbyteorder() or \
           h.dtype == header_image_dtype or h.dtype == header_image_dtype.newbyteorder()

def is_readable(filename):
    ''' Test if the file read has a valid MRC header
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    
    :Returns:
        
    out : bool
          True if the header conforms to MRC
    '''
    
    if hasattr(filename, 'dtype'): 
        h = filename
        if not is_format_header(h):
            raise ValueError, "Array dtype incorrect"
    else: 
        try:
            h = read_header(filename)
        except: return False
    if h['mode'][0] not in mrc2numpy: return False
    if h['byteorder'][0] not in intbyteorder and \
       h['byteorder'][0].byteswap() not in intbyteorder: return False
    if not numpy.alltrue([h[v][0] > 0 for v in ('nx', 'ny', 'nz', 'mx', 'my', 'mz')]): return False
    return True

def read_header(filename, index=None):
    ''' Read the MRC header
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    index : int, ignored
            Index of image to get the header, if None, the stack header (Default: None)
    
    :Returns:
        
    out : array
          Array with header information in the file
    '''
    
    f = _open(filename, 'r')
    try:
        curr = f.tell()
        h = numpy.fromfile(f, dtype=header_image_dtype, count=1)
        if not is_readable(h):
            f.seek(curr)
            h = numpy.fromfile(f, dtype=header_image_dtype.newbyteorder(), count=1)
        if not is_readable(h): raise IOError, "Not an MRC file"
    finally:
        _close(filename, f)
    return h

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
    else: h = read_header(filename)
    return h['nz']

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
    
    f = _open(filename, 'r')
    try:
        if index is None: index = 0
        h = read_header(f)
        count = count_images(h)
        if header is not None:  _update_header(header, h, mrc2ara, 'mrc')
        d_len = h['nx'][0]*h['ny'][0]
        dtype = numpy.dtype(mrc2numpy[h['mode'][0]])
        offset = 1024 + index * d_len * dtype.itemsize;
        f.seek(offset)
        for i in xrange(index, count):
            out = numpy.fromfile(f, dtype=dtype, count=d_len)
            if index is None and int(h['nz'][0]) > 1: out = out.reshape(int(h['nx'][0]), int(h['ny'][0]), int(h['nz'][0]))
            elif int(h['ny'][0]) > 1: out = out.reshape(int(h['nx'][0]), int(h['ny'][0]))
            yield out
    finally:
        _close(filename, f)

def read_image(filename, index=None, header=None):
    ''' Read an image from the specified file in the MRC format
    
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
    
    f = _open(filename, 'r')
    try:
        if index is None: index = 0
        h = read_header(f)
        if header is not None: _update_header(header, h, mrc2ara, 'mrc')
        count = count_images(h)
        if index >= count: raise IOError, "Index exceeds number of images in stack: %d < %d"%(index, count)
        d_len = h['nx'][0]*h['ny'][0]
        dtype = numpy.dtype(mrc2numpy[h['mode'][0]])
        offset = 1024 + index * d_len * dtype.itemsize;
        f.seek(offset)
        out = numpy.fromfile(f, dtype=dtype, count=d_len)
        if index is None and int(h['nz'][0]) > 1:   out = out.reshape(int(h['nx'][0]), int(h['ny'][0]), int(h['nz'][0]))
        elif int(h['ny']) > 1: out = out.reshape(int(h['nx'][0]), int(h['ny'][0]))
    finally:
        _close(filename, f)
    return out

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
    
    try: img = img.astype(mrc2numpy[numpy2mrc[img.dtype.type]])
    except:
        raise TypeError, "Unsupported type for MRC writing: %s"%str(img.dtype)
    
    f = _open(filename, 'w')
    if header is None or not hasattr(header, 'dtype') or not is_format_header(header):
        h = numpy.zeros(1, header_image_dtype) if index is None else numpy.zeros(1, header_stack_dtype)
        _update_header(h, mrc_defaults, ara2mrc)
        pix = header.get('apix', 1.0) if header is not None else 1.0
        header=_update_header(h, header, ara2mrc, 'mrc')
        header['nx'] = img.shape[-1]
        header['ny'] = img.shape[-2] if img.ndim > 1 else 1
        if header['nz'] == 0:
            header['nz'] = img.shape[-3] if img.ndim > 2 else 1
        header['mode'] = numpy2mrc[img.dtype.type]
        header['mx'] = header['nx']
        header['my'] = header['ny']
        header['mz'] = header['nz']
        header['xlen'] = header['nx']*pix
        header['ylen'] = header['ny']*pix
        header['zlen'] = header['nz']*pix
        header['amin'] = numpy.min(img)
        header['amax'] = numpy.max(img)
        header['amean'] = numpy.mean(img)
        header['rms'] = numpy.std(img)
        if img.ndim == 3:
            header['nxstart'] = header['nx'] / -2
            header['nystart'] = header['ny'] / -2
            header['nzstart'] = header['nz'] / -2
        if index is not None:
            stack_count = header['nz'] if header['nz'] > 1 else index+1
            header['nz'] = stack_count
            header['mz'] = stack_count
            header['zlen'] = stack_count
            header['zorigin'] = stack_count/2.0
            header['nsymbt'] = stack_count * 88
            header['nintegers'] = 0
            header['nfloats'] = 22
    
    offset=0
    try:
        if f != filename:
            header.tofile(f)
            if index > 0: f.seek(offset+index*img.ravel().shape[0])
        img.tofile(f)
    finally:
        _close(filename, f)

if __name__ == '__main__':
    
    print len(header_image_dtype)
    print header_stack_dtype.itemsize
