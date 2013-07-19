'''  Defines a set of utility functions

.. Created on Jul 18, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy

def open(filename, mode):
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

def close(filename, fd):
    ''' Close the file descriptor (if it was opened by caller)
    
    filename : str
               Name of the file
    fd : File
         File descriptor
    '''
    
    if fd != filename: fd.close()

def update_header(dest, source, header_map, tag=None):
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

def read_image(f, dtype, dlen, shape, swap, order='C'):
    ''' Read an image from a file using random file acess
    
    :Parameters:
    
    f : stream
        Input file stream
    dtype : dtype
            Data type 
    dlen : int
           Number of elements
    shape : tuple
            Shape of the array 
    swap : bool
           Swap the byte order
    order : str
            Layout of a 2 or 3D array
    
    :Returns:
    
    out : ndarray
          Array of image data
    '''
    
    out = numpy.fromfile(f, dtype=dtype, count=dlen)
    out.shape = shape
    out = out.squeeze()
    if order == 'F':
        out.shape = out.shape[::-1]
        out = out.transpose()
    if swap:out = out.byteswap().newbyteorder()
    return out
