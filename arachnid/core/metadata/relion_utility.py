''' Set of utilities to deal with RELION star files

This module provides a set a utility functions to handle RELION star files.

.. Created on Aug 30, 2013
.. codeauthor:: robertlanglois
'''
import spider_utility
import os

def replace_filename(data, newfilename=None, reindex=False):
    ''' Replace the image file name, reindex the stack indices or both
    
    :Parameters:
    
    data : list
           List of namedtuples read from a Relion star file
    newfilename : str
                  New filename for the image name column
    reindex : bool
              If true, renumber the particles in each stack consecutively
              
    :Returns:
    
    out : list
           List of updated namedtuples
    '''
    
    idmap={} if reindex else None
    newdata=[]
    for d in data:
        filename, id = relion_id(d.rlnImageName, 0, False)
        if newfilename is not None:
            if spider_utility.is_spider_filename(newfilename):
                filename=spider_utility.spider_filename(newfilename, filename)
            else: filename=newfilename
        if idmap is not None:
            if filename not in idmap: idmap[filename]=0
            idmap[filename] += 1
            id = idmap[filename]
        newdata.append(d._replace(rlnImageName=relion_filename(filename, id)))
    return newdata
    

def relion_filename(filename, id):
    '''Extract the filename and stack index
    
    This function extracts the spider ID as an integer.
        
    .. sourcecode:: py
    
        >>> relion_id("0001@basename00010.ext")
        (basename00010.ext, 1)

    :Parameters:
    
    filename : str
               A file name
    
    :Returns:
        
    return_val : tuple 
                 Micrograph ID, particle ID or Micrograph ID only
    '''
    
    pid=""
    try:int(id)
    except:
        if id.find('@') != -1:
            pid,id = id.split('@')
        if spider_utility.is_spider_filename(filename):
            return pid+"@"+spider_utility.spider_filename(filename, id)
    else: pid = str(id)
    return pid+"@"+filename

def frame_filename(filename, id):
    ''' Create a frame file name from a template and an ID
    
    :Parameters:
    
    filename : str
               A file name
    
    :Returns:
    
    '''
    
    base = os.path.basename(filename)
    pos = base.find('_')+1
    if pos == -1: raise ValueError, "Not a valid frame filename"
    end = base.find('_', pos)
    if end == -1: raise ValueError, "Not a valid frame filename"
    return os.path.join(os.path.dirname(filename), base[:pos]+str(id)+base[end:])

def relion_file(filename, file_only=False):
    '''Extract the filename and stack index
    
    This function extracts the spider ID as an integer.
        
    .. sourcecode:: py
    
        >>> relion_id("0001@basename00010.ext")
        (basename00010.ext, 1)

    :Parameters:
    
    filename : str
               A file name
    
    :Returns:
        
    return_val : tuple 
                  Micrograph ID, particle ID or Micrograph ID only
    '''
    
    if filename.find('@') != -1:
        pid,mid = filename.split('@')
        if file_only: return mid
        pid = int(pid)
        return (mid, pid)
    return filename

def relion_id(filename, idlen=0, use_int=True):
    '''Extract the Spider ID as an integer
    
    This function extracts the spider ID as an integer.
        
    .. sourcecode:: py
    
        >>> relion_id("0001@basename00010.ext")
        (1, 10)

    :Parameters:

    filename : str
               A file name
    idlen : int 
            Maximum length of ID (default 0)
    use_int : bool
             Convert to integer, (default True)
    
    :Returns:
    
    return_val : Tuple 
                 Micrograph ID, particle ID or Micrograph ID only
    '''
    
    if filename.find('@') != -1:
        pid,mid = filename.split('@')
        if use_int: pid = int(pid)
        try:
            return (spider_utility.spider_id(mid, idlen, use_int), pid)
        except: return (None, pid)
    return (spider_utility.spider_id(filename, idlen, use_int), None)



