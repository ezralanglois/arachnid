''' Set of utilities to deal with SPIDER files

This module provides a set a utility functions to handle spider files.

.. Created on Sep 28, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from format_utility import split_id, object_id, fileid, has_file_id
from type_utility import is_int
from collections import namedtuple, defaultdict
import re, os, logging
import numpy

__spider_identifer = namedtuple("SpiderIdentifer", "fid,id")

def images2label(images):
    ''' Convert an image list to an ID label array
    '''
    
    label = numpy.zeros((len(images), 2))
    for i in xrange(len(images)):
        label[i, :] = (spider_id(images[i][0]), images[i][1])
    return label

def single_images(files):
    ''' Organize a set of files into groups of single images
    
    :Parameters:
    
    files : list
            List of filenames of form: 40S_001_001.spi
    
    :Returns:
    
    groups : list
             List of tuples {1: ['40S_001_001.spi' ...]}
    '''
    
    groups = defaultdict(list)
    for f in files:
        id = spider_basename(f)
        try:
            if id[-1]=='-' or id[-1]=='_': id=id[:len(id)-1]
            id = spider_id(id)
            if id == 0 or isinstance(id, str):
                id = spider_basename(f)
                if id[-1]=='-' or id[-1]=='_': id=id[:len(id)-1]
                id = id.replace('.', '0')
                try:
                    id = spider_id(id)
                except: id=""
                if isinstance(id, str):
                    n=id.find('_')
                    if n != -1: id = id[:n]
                    id = spider_id(id)
                    
                
        except: pass
        groups[id].append(f)
    
    for key in groups.iterkeys():
        idx = numpy.argsort([spider_id(id) for id in groups[key]])
        groups[key] = [groups[key][i] for i in idx]
        
    return groups.items()
    
def select_subset(files, select, id_len=0):
    ''' Create a list of files based on the given selection
    
    :Parameters:
    
    files : list
            List of filenames
    select : array
             Array of file ids
    id_len : int
             Maximum length of SPIDER id
             
    :Returns:
    
    out : list
          List of selected filenames
    '''
    
    if len(select) == 0: return []
    if hasattr(select[0], 'select'):
        return [spider_filename(files[0], s.id) for s in select if s.select > 0]
    elif hasattr(select[0], 'id'):
        return [spider_filename(files[0], s.id) for s in select]
    else:
        return [spider_filename(files[0], s[0]) for s in select]

def update_spider_files(map, id, *files):
    ''' Update the list of files in the dictionary to the current ID
    
    :Parameters:
    
    map : dict
          Dictionary mapping file param to value
    id : int
         SPIDER ID
    files : list
            List of files to update
    '''
    
    for key in files:
        if map[key] != "" and is_spider_filename(map[key]):
            header = ""
            if map[key].find('=') != -1:
                map[key], header = map[key].split('=')
                header = '='+header
            try:
                map[key] = spider_filename(map[key], id)+header
            except:
                logging.error("key=%s"%key)
                raise

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
        if is_spider_filename(filename):
            return pid+"@"+spider_filename(filename, id)
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
            return (spider_id(mid, idlen, use_int), pid)
        except: return (None, pid)
    return (spider_id(filename, idlen, use_int), None)

def spider_header_vals(line):
    '''Parse the spider header into a set of words
    
    This function uses regular expressions to parse a Spider header.
        
    .. sourcecode:: py
    
        >>> from core.metadata.spider_utility import *
        >>> spider_header_vals(";          X           Y       PARTICLE NO.   PEAK HT")
        ['X', 'Y', 'PARTICLE NO', 'PEAK HT']
    
    :Parameters:

    line : str
           A string header
        
    :Returns:
        
    return_val : list
                A list of string header values
    '''
    
    line = line[1:].strip()
    vals = re.split("\W\W+", line)
    index = 0
    while index < len(vals):
        if vals[index] == '':
            del vals[index]
        else:
            index = index + 1
    return vals

def spider_header_vars(line):
    '''Parse the spider header into a set of variable compliant names
    
    This function parses a Spider header into a list of variable compliant names.
        
    .. sourcecode:: py
    
        >>> spider_header_vars(";          X           Y       PARTICLE NO.   PEAK HT")
        ['x', 'y', 'particle_no', 'peak_ht']
    
    :Parameters:

    line : str
           A string header
    
    :Returns:

    return_val : list
                A list of string header values
    '''
    
    header = spider_header_vals(line)
    for i in range(len(header)):
        val = header[i].lower().replace(".", "")
        header[i] = "_".join(val.split())
        if header[i] == "class": header[i] = "select"
    return header

def spider_id_length(base):
    '''Count the number of integers at the end of a string name
    
    This function counts the the number of integers in a base filename.
        
    .. sourcecode:: py
    
        >>> spider_id_length("path/basename00001")
        5
    
    :Parameters:

    base : str
          A file base name
    
    :Returns:
    
    return_val : int 
                 Length of spider id
    '''
    
    maxlen = 0
    for ch in reversed(base):
        try:
            int(ch)
        except ValueError: break
        maxlen = maxlen + 1
    return maxlen

def is_spider_filename(filename):
    '''Test if the filename conforms to a Spider filename
    
    This function tests if a filename conforms to a Spider filename, e.g. basename00001.ext.
        
    .. sourcecode:: py
    
        >>> is_spider_filename("path/basename00001.ext")
        True
        >>> is_spider_filename("path/basename.ext")
        False
    
    :Parameters:

    filename : str
               A file name
    
    :Returns:
    
    return_val : bool 
                 True if filename conforms to Spider filename
    '''
    
    try: int(filename)
    except: pass
    else: return True
    
    try: '+'+filename
    except:
        for f in filename:
            if not is_spider_filename(f): return False
        return True
    
    filename, ext = os.path.splitext(filename)
    return spider_id_length(filename) > 0

def spider_basename(filename, idlen=0):
    '''Extract the basename of a spider file (no ID)
    
    This function extracts the basename of a spider file, without the ID.
        
    .. sourcecode:: py
    
        >>> spider_basename("path/basename00001.ext")
        "basename"

    :Parameters:
    
    filename : str
               A file name
    idlen : int 
            Maximum length of ID (default 0)
    
    :Returns:
    
    return_val : str 
                 Base spider filename
    '''
    
    filename = os.path.splitext(os.path.basename(filename))[0]
    n = spider_id_length(filename)
    if n > 0:
        if n > idlen: idlen = n
        filename = filename[:len(filename)-idlen]
    return filename

def spider_searchpath(filename, wildcard='*', idlen=0):
    '''Replace SPIDER ID with wild card
    
    This function extracts the basename and extension of a spider file, without the ID.
        
    .. sourcecode:: py
    
        >>> spider_filepath("path/basename00001.ext")
        "path/basename*.ext"

    :Parameters:

    filename : str
               A file name
    wildcard : str
               Wild card to substitute for spider ID
    idlen : int 
            Maximum length of ID (default 0)
    
    :Returns:
    
    return_val : str 
                 Base spider filename with wildcard in place of ID
    '''
    
    filename, ext = os.path.splitext(filename)
    n = spider_id_length(filename)
    if n > 0:
        if n > idlen: idlen = n
        filename = filename[:len(filename)-idlen]
    return filename+wildcard+ext

def spider_filepath(filename, idlen=0):
    '''Extract the basename of a spider file (no ID) with extension
    
    This function extracts the basename and extension of a spider file, without the ID.
        
    .. sourcecode:: py
    
        >>> spider_filepath("path/basename00001.ext")
        "path/basename.ext"

    :Parameters:

    filename : str
               A file name
    idlen : int 
            Maximum length of ID (default 0)
    
    :Returns:
    
    return_val : str 
                 Base spider filename with extension
    '''
    
    filename, ext = os.path.splitext(filename)
    n = spider_id_length(filename)
    if n > 0:
        if n > idlen: idlen = n
        filename = filename[:len(filename)-idlen]
    return filename+ext

def spider_id(filename, idlen=0, use_int=True):
    '''Extract the Spider ID as an integer
    
    This function extracts the spider ID as an integer.
        
    .. sourcecode:: py
    
        >>> spider_id("basename00010.ext")
        10

    :Parameters:

    filename : str
               A file name
    idlen : int 
            Maximum length of ID (default 0)
    use_int: bool
             Convert to integer, (default True)
    
    :Returns:
    
    return_val : int 
                 Spider ID
    '''
    
    try:
        return int(filename)
    except: pass
    
    orig = filename
    if filename.find('.') != -1:
        filename, ext = os.path.splitext(filename)
    n = spider_id_length(filename)
    if n < idlen or idlen == 0: idlen = n
    try:
        filename = filename[len(filename)-idlen:]
        val = int(filename)
        if use_int: return val
        return filename
    except:
        raise SpiderError, "Cannot parse filename: "+orig+" - "+str(len(filename))+" - "+str(idlen)

def split_spider_id(id, idlen=0, use_int=True):
    '''Extract a concatenated Spider ID as a tuple of integers
    
    This function extracts a tuple of integers from concatenated spider ID
        
    .. sourcecode:: py
    
        >>> split_spider_id("basename00001.ext/10")
        SpiderIdentifer(fid=1, id=10)

    :Parameters:

    id : str
         Concatenated identifier (or a list of identifiers)
    idlen : int 
            Maximum length of ID (default 0)
    use_int: bool
             Convert to integer, (default True)
    
    :Returns:
    
    return_val : tuple 
                 (File ID, Object ID) or list of tuples
    '''
    
    if isinstance(id, list):
        vals = []
        for i in id:
            vals.append(split_spider_id(i, idlen, use_int))
        return vals
    if hasattr(id, "id"): id = id.id
    try:
        fid, oid = split_id(id)
        return __spider_identifer(spider_id(fid, idlen, use_int), object_id(oid))
    except:
        return object_id(id)

def tuple2id(tvals, filename=None, id_len=0):
    ''' Convert the id field in an array of named tuples to
    a 2 or 1-D numpy array.
    
    :Parameters:
        
    tvals : list
            List of namedtuples
    filename : str
               Name of the micrograph file
    id_len : int
             Maximum length of the filename id
    
    :Returns:
        
        vals : array
               1 or 2-dimensional array of ids
    '''
    
    if has_file_id(tvals):
        vals = numpy.zeros((len(tvals), 2))
        for i in xrange(len(vals)):
            vals[i, :] = split_id(tvals[i].id, True)
    elif filename is not None:
        vals = numpy.zeros((len(tvals), 2))
        fid = spider_id(fileid(filename), id_len, True)
        for i in xrange(len(vals)):
            vals[i, :] = (fid, object_id(tvals[i].id))
    else:
        vals = numpy.zeros((len(tvals), 1))
        for i in xrange(len(vals)):
            vals[i, 0] = object_id(tvals[i].id)
    return vals

def spider_template(filename, template, idlen=0):
    ''' Generate a new spider filename with the same id length and extension
    
    :Parameters:
    
    filename : str
               Path and base name for new SPIDER template
    template : str
               SPIDER template
    idlen : int
            ID length to use
    
    :Returns:
    
    filename : str
               New SPIDER template
    '''
    
    if filename.find('.') != -1:
        filename = os.path.splitext(filename)[0]
    ext = os.path.splitext(template)[1]
    if idlen == 0: idlen = spider_id_length(os.path.splitext(template)[0])
    filename+="0".zfill(idlen)
    return filename + ext

def add_spider_id(filename, idlen):
    ''' Add an empty spider ID (all zeros) to the end of a filename
    
    :Parameters:
    
    filename : str
               Path and base name for new SPIDER template
    idlen : int
            ID length to use
    
    :Returns:
    
    filename : str
               New SPIDER template
    '''
    
    filename, ext = os.path.splitext(filename)
    filename+="0".zfill(idlen)
    return filename + ext
    

def spider_filename(filename, id, idlen=0):
    '''Create a Spider filename with given filepath and ID
    
    This function creates a spider filename with given filepath and ID.
        
    .. sourcecode:: py
    
        >>> spider_filename("basename00000.ext", 1)
        "basename00001.ext"

    :Parameters:

    filename : str
               A file name
    id : int or string
         Spider ID
    idlen : int 
            Maximum length of ID (default 0)
    
    :Returns:
    
    return_val : str
                 A new spider file name
    '''
    
    if filename.find('=') != -1:
        filename = filename.split('=', 1)[0]
    
    if not is_spider_filename(filename):
        filename, ext = os.path.splitext(filename)
        id = extract_id(id)
        return filename+str(id)+ext
    filename, ext = os.path.splitext(filename)
    n = spider_id_length(filename)
    if idlen == 0 or n < idlen: idlen = n
    if not is_int(id):
        id = os.path.splitext(id)[0]
        n = spider_id_length(id)
        #if n < idlen: n=idlen
        #idn = id[len(id)-n:]
        if n < idlen: idlen=n
        idn = id[len(id)-idlen:]
    else: idn = id
    try:
        id = int(float(idn))
    except:
        return filename+str(spider_id(id, idlen)).zfill(idlen)+ext
    return filename[:len(filename)-idlen]+str(id).zfill(idlen)+ext

def file_map(filename, selection, id_len=0, ensure_exist=False):
    ''' Create a map between SPIDER ids and SPIDER filenames
    
    :Parameters:

    filename : str
               A file name
    id : int or string
         Spider ID
    idlen : int 
            Maximum length of ID (default 0)
    
    :Returns:
    
    filemap : dict
             Dictionary of SPIDER ID, SPIDER filename pairs
    '''
    
    try: selection = int(selection)
    except: file_ids = numpy.unique(selection.astype(numpy.int))
    else: file_ids = xrange(1, selection+1)
    filemap = {}
    for id in file_ids:
        filemap[id] = spider_filename(filename, id, id_len)
        if ensure_exist:
            if not os.path.exists(filemap[id]):
                raise IOError, "File does not exist error: %s"%filemap[id]
    return filemap

def extract_id(filename, id_len=0):
    '''Extract a minimal identifier from the processed filename
        
    .. sourcecode:: py
    
        >>> extract_id("basename00010.ext")
        10
        >>> extract_id("basename.ext")
        'basename'
    
    :Parameters:
    
    filename : str
               Processed filename
    id_len : int
             Maximum length of integer ID
    
    :Returns:
        
    id : str
         ID extracted from a filename
    '''
    
    try: return spider_id(filename, id_len, False)
    except: return os.path.splitext(os.path.basename(filename))[0]

class SpiderError(ValueError):
    """Exception raised for errors in spider ID parsing
    """
    pass

def test_valid_spider_input(input_files):
    ''' Test if list of multiple files all contain SPIDER IDs
    
    :Parameters:
    
    input_files : list
                  List of input files
    
    :Returns:
    
    valid : bool
            True if only one file in list or every file as a SPIDER ID
    '''
    
    if len(input_files) > 1:
        for f in input_files:
            if not is_spider_filename(f):
                return False
    return True
        

