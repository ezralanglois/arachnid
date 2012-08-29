''' Convert a Python variable to the proper format for a SPIDER parameter

.. Created on Aug 10, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import spider_var
import os

def is_incore_filename(filename):
    '''Test if the given filename refers to a spider incore file
    
    :Parameters:
        
    filename : str
               Name of the file
                   
    :Returns:
        
    is_incore : bool
                True if the filename refers to a spider incore file
    '''
    
    if isinstance(filename, tuple): 
        filename = filename[0]
    return isinstance(filename, spider_var.spider_var) or isinstance(filename, int)

def spider_doc(filename):
    '''Ensure correct format for spider file or incore file
    
    :Parameters:
    
    filename : str
               Name of the file
                   
    :Returns:
    
    filename : str
               Correct spider filename
    '''
    
    if filename == "" or filename is None: return "*"
    if is_incore_filename(filename): filename = "s%d_"%filename
    elif filename.rfind('.') != -1:
        if len(os.path.splitext(filename)[1][1:]) == 3:
            filename = os.path.splitext(filename)[0]
    return filename

def spider_image(filename, index=None):
    ''' Ensure correct format for a SPIDER image filename
    
    :Parameters:
        
    filename : str
                Name of the file
    index : int, optional
            Index of image in a stack (if None, then assume single image)
          
    :Returns:
    
    filename : str
               Correct spider filename
    
    '''
    
    if isinstance(filename, tuple): filename, index = filename
    if filename == "" or filename is None: return "*"
    filename = spider_doc(filename)
    if index is not None:
        if filename[-1] == '@': filename += str(index)
        else: filename += '@'+str(index)
    return filename

def spider_stack(filename, size=None):
    ''' Ensure correct format for a SPIDER stack filename
    
    .. todo:: handle series of single images
    
    :Parameters:
        
    filename : str
                Name of the file
    size : int, optional
           Size of the spider stack (if None, then assume whole stack)
          
    :Returns:
    
    filename : str
               Correct spider filename
    '''
    
    if filename == "" or filename is None: return "*"
    filename = spider_doc(filename)
    if filename.find('@') == -1: filename += '@'
    if size is not None and size > 0 and filename[-1] != '*':
        filename = filename.rjust(len(str(size)), '*')
    return filename

def spider_select(filename):
    '''Provide a properly formated range or selection filename
    
    :Parameters:
        
    filename : object
               Range object: int, tuple of ints, selection filename
                   
    :Returns:
        
    range : str
            Range or filename
    '''
    
    if isinstance(filename, int) and not isinstance(filename, spider_var.spider_var):
        return "(1-%d)"%(filename)
    elif isinstance(filename, tuple) and len(filename) == 2 and isinstance(filename[0], int) and isinstance(filename[1], int):
        return "(%d-%d)"%filename
    return spider_doc(filename)

def spider_coord_tuple(values):
    ''' Create a Spider coordinate tuple for either size or position in 3D
    
    :Parameters:
    
    values : tuple
             Integer, 2-tuple or 3-tuple
    
    :Returns:
    
    val : str
          Properly formatted SPIDER tuple
    '''
    
    if isinstance(values, tuple):
        x_size, y_size = values[0], values[1]
        z_size = values[2] if len(values) > 2 else 1
    else: x_size, y_size, z_size = values, values, 1
    return spider_tuple(x_size, y_size, z_size)

def spider_tuple(*vals, **kwargs):
    '''Test if the given filename refers to a spider incore file
    
    Undocumented keywords arguments supported:
        
        #. `int_format` : format string for an integer (Default: %d)
        #. `float_format` : format string for a float (Default: %f)
    
    :Parameters:
    
    vals : tuple
           Group of spider parameters in numeric format, e.g. int or float
    kwargs : dict
             Dictionary of key word arguments: either float_format or int_format
                   
    :Returns:
    
    formatted : str
                Set of numbers correctly formated in a comma-separated string
    '''
    
    float_format = kwargs['float_format'] if 'float_format' in kwargs else '%f'
    int_format = kwargs['int_format'] if 'int_format' in kwargs else '%d'
    vals = list(vals)
    for i in xrange(len(vals)):
        if isinstance(vals[i], bool):
            vals[i] = int_format%vals[i]
        elif isinstance(vals[i], int):
            vals[i] = int_format%vals[i]
        elif isinstance(vals[i], float):
            vals[i] = float_format%vals[i]
        else:
            raise SpiderParameterError, "Requires int or float values: %s"%str(vals[i])
    return ",".join(vals)

class SpiderParameterError(StandardError):
    ''' Exception is raised when there is an error converting a
    Python value to a SPIDER parameter.
    '''
    pass
