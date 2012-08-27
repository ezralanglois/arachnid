'''

.. seealso::

    http://www.wadsworth.org/spider_doc/spider/docs/image_doc.html
    

.. Created on Aug 9, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from arachnid.core.metadata import type_utility
from .. import ndimage
import numpy

header = {'nz': 1,'ny': 2,'irec':3,'iform':5,'imami':6,'fmax':7,'fmin':8, 
          'av':9, 'sig':10, 'nx':12, 'labrec':13, 'labbyt': 22, 'lenbyt':23, 
          'istack': 24, 'maxim':26, 'imgnum': 27, 'apix':38,'voltage': 39,
          'proj': 40, 'mic': 41}

def _create_header(vtype='<f4'):
    idx2header = dict([(val, key) for key, val in header.iteritems()])
    return numpy.dtype([(idx2header.get(i, "unused_%s"%str(i+1).zfill(2)), vtype) for i in xrange(1, numpy.max(header.values())+1)])

header_dtype_le = _create_header('<f4')
header_dtype_be = _create_header('>f4')

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
        if h.dtype != header_dtype_le and h.dtype != header_dtype_be: 
            raise ValueError, "Array dtype incorrect"
    else: h = read_spider_header(filename)
    for i in ('nz','ny','iform','nx','labrec','labbyt','lenbyt'):
        if not type_utility.is_float_int(h[i]): return False
    if not int(h['iform']) in (1,3,-11,-12,-21,-22): return False
    if (int(h['labrec'])*int(h['lenbyt'])) != int(h['labbyt']): return False
    return True

def read_header(filename, index=None):
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
    
    h = read_spider_header(filename, index)
    header = numpy.zeros(1, dtype=ndimage._header)
    header[0].apix = h['apix']
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
    
    try: "+"+filename
    except: f = filename
    else:  f = open(filename, 'r')
    try:
        header_dtype = header_dtype_le
        curr = f.tell()
        h = numpy.fromfile(f, dtype=header_dtype, count=1)
        if not is_readable(h):
            header_dtype = header_dtype_be
            f.seek(curr)
            h = numpy.fromfile(f, dtype=header_dtype, count=1)
        if not is_readable(h): raise IOError, "Not a SPIDER file"
        if index is not None:
            h_len = int(h['labbyt'])
            i_len = int(h['nx']) * int(h['ny']) * int(h['nz']) * 4
            count = max(int(h['istack']), 1)
            if index >= count: raise IOError, "Index exceeds number of images in stack: %d < %d"%(index, count)
            offset = index * (h_len+i_len)
            f.seek(offset)
            h = numpy.fromfile(f, dtype=header_dtype, count=1)
    finally:
        if f != filename: f.close()
    return h

def read_image(filename, index=None):
    ''' Read a SPIDER image
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    index : int, optional
            Index of image to get, if None, first image (Default: None)
    
    :Returns:
        
    out : array
          Array with header information in the file
    '''
    try: "+"+filename
    except: f = filename
    else:  f = open(filename, 'r')
    try:
        if index is None: index = 0
        h = read_spider_header(f)
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
        if f != filename: f.close()
    return out

def iter_images(filename, index=None):
    ''' Read a set of SPIDER images
    
    :Parameters:
    
    filename : str or file object
               Filename or open stream for a file
    index : int, optional
            Index of image to start, if None, start with the first image (Default: None)
    
    :Returns:
        
    out : array
          Array with header information in the file
    '''
    
    try: "+"+filename
    except: f = filename
    else:  f = open(filename, 'r')
    try:
        if index is None: index = 0
        h = read_spider_header(f)
        h_len = int(h['labbyt'])
        d_len = int(h['nx']) * int(h['ny']) * int(h['nz'])
        i_len = d_len * 4
        count = count_images(h)
        if index >= count: raise IOError, "Index exceeds number of images in stack: %d < %d"%(index, count)
        offset = h_len + index * (h_len+i_len)
        f.seek(offset)
        for i in xrange(index, count):
            out = numpy.fromfile(f, dtype=h.dtype.fields['nx'][0], count=d_len)
            if int(h['nz']) > 1:   out = out.reshape(int(h['nx']), int(h['ny']), int(h['nz']))
            elif int(h['ny']) > 1: out = out.reshape(int(h['nx']), int(h['ny']))
            yield out
    finally:
        if f != filename: f.close()

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
    else: h = read_spider_header(filename)
    return max(int(h['istack']), 1)

def write_header(filename, img, header=None):
    '''
    '''
    
    pass

def write_image(filename, img, index=None, header=None):
    '''
    '''
    
    pass
