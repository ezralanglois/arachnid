''' Read and write into a number of supported document formats

A number of different document formats are supported including:

    - Comma Separated Value Format
    
       .. automodule:: arachnid.core.metadata.formats.csv
           :noindex:
       
    - Malibu Prediction Format
    
       .. automodule:: arachnid.core.metadata.formats.prediction
           :noindex:
        
    - Spider Document Format
    
        .. automodule:: arachnid.core.metadata.formats.spiderdoc
           :noindex:
        
    - Spider Selection Format
        
        .. automodule:: arachnid.core.metadata.formats.spidersel
           :noindex:
        
    - STAR Format
        
        .. automodule:: arachnid.core.metadata.formats.star
           :noindex:

The format of a document will be automatically determined when the
document is read.

When writing out a document, the format will be chosen based on the extension.

.. Created on Jun 8, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from formats import csv, prediction, project, spiderdoc, spidersel, star
from format_utility import ParseFormatError, WriteFormatError
from spider_utility import spider_filename, split_spider_id
from ..parallel import mpi_utility
import format_utility
import os, numpy, logging

__formats = [star, spiderdoc, spidersel, project, csv, prediction]
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
#_logger.setLevel(logging.DEBUG)

def filters(formats=None):
    '''Get a list file filters from a list of supported formats
    
    This filters are in a format compatible with PyQT4 FileDialogs.
    
    .. sourcecode:: py
    
        >>> from arachnid.core.metadata.format import *
        >>> filters()
        "Project (*.prj);;CSV (*.csv);;Prediction (*.pred);;Spider Doc File (*.dat);;Spider Selection Doc (*.sdat)"
    
    .. note ::
    
        Due to a "bug" in Sphinx, the default value for the `formats` parameter in each
        function is None: when formats is None, its is automatically set to use
        all available formats as defined in the module variable `__formats`.
    
    :Parameters:
    
    formats : list
              List of formats, default All
    
    :Returns:
    
    val : string
          Formated list of format filters
    '''
    
    if formats is None: # Ensure project is the first format when loading in a file chooser dialog
        formats=__formats[1:]
        formats.append(formats[0])
    return ";;".join([f.filter() for f in formats])

def extension(filter, formats=None):
    '''Get the extension associated with the given file filter
    
    .. sourcecode:: py
    
        >>> from arachnid.core.metadata.format import *
        >>> extension("Project (*.prj)")
        "prj"
        >>> extension("Prediction (*.pred)")
        "pred"
    
    :Parameters:
    
    filter : string
             File filter to test
    formats : list
              List of formats, default All
    
    :Returns:
    
    val : string
          Extension associated with given filter
    '''
    
    if formats is None: formats=__formats
    for f in formats:
        if f.filter() == filter: return f.extension()
    raise ValueError, "Cannot find extension for filter: "+filter

def replace_extension(filename, filter, formats=None):
    '''Replace the extension of the file given the filter
    
    :Parameters:
    
    filename : str
               Name of the file
    filter : str
             Name of the file filter
    formats : list, optional
             List of available formats
    
    :Returns:
    
    filename : str
               Filename with new extension
    '''
    
    filename = os.path.splitext(str(filename))[0]
    return filename + "." + extension(filter, formats)

def open_file(filename, mode='r', header=None, spiderid=None, id_len=0, prefix=None, replace_ext=False, **extra):
    '''Get the extension associated with the given file filter
    
    This functions removes the header (if there is one) at the end of the string
    and if specified, creates a new filename using the given template and new ID.
    
    .. sourcecode:: py
         
        >>> from arachnid.core.metadata.format import *
        >>> from collections import namedtuple
        >>> BasicTuple = namedtuple("BasicTuple", "id,x,y")
        >>> values = [ BasicTuple("1", 572.00, 228.00 ), BasicTuple("2", 738.00, 144.00 ), BasicTuple("3", 810.00, 298.00) ]
        >>> filename, fout = open_file("doc001.spi", mode='w')
        >>> spiderdoc.write(fout, values)
        >>> fout.close()
        >>> import os
        >>> os.system("more doc001.spi")
        ;            id          x          y
        1  31         572         228
        2  32         738         144
        3  33         810         298

        >>> filename, fin, header = open_file("doc001.spi=id;x;y")
        >>> filename
        "doc001.spi"
        >>> header
        ['id', 'x', 'y']
        >>> spiderdoc.read_header(fin, header)
        (<bound method type._make of <class 'core.metadata.factories.namedtuple_factory.BasicTuple'>>, '1  31         572         228')
        >>> header
        ['id', 'x', 'y']
    
    :Parameters:
    
    filename : string
             Filename template used to create new filename
    header : list
             List of string values
    spiderid : id-like object
              Filename, string or integer to use as an ID
    id_len : integer
             Maximum ID length
    prefix : str, optional
             Prefix to add to start of the filename
    replace_ext : bool
                  If True and spiderid is a filename with an extension, replace current filename
                  with this extension.
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
    
    val : string
          New filename without header and with new ID
    '''
    
    _logger.debug("open_file: "+str(filename)+" - mode: %s"%(mode))
    if header is not None: _logger.debug("Using initial header: %s"%str(header))
    filename, header = format_utility.parse_header(filename, header)
    _logger.debug("Using header: %s"%str(header))
    if spiderid is not None: 
        filename = spider_filename(filename, spiderid, id_len)
    filename = format_utility.add_prefix(filename, prefix)
    if replace_ext and isinstance(spiderid, str):
        ext = os.path.splitext(spiderid)[1]
        if ext != "":
            base = os.path.splitext(filename)[0]
            filename = base+ext
    filename = os.path.expanduser(filename)
    try:
        fin = open(filename, mode)
    except:
        _logger.debug("Error opening "+filename+" - "+str(os.path.exists(filename)))
        raise
    if mode != 'r': return filename, fin
    return filename, fin, header

def get_format_by_ext(filename, format=None, formats=None, default_format=None, **extra):
    '''Find appropriate format for a filename using the file extension
    
    .. sourcecode:: py
                 
        >>> from arachnid.core.metadata.format import *
        >>> format = get_format_by_ext("data.dat")
        >>> format.extension()
        'dat'
        >>> format.__name__
        'core.metadata.formats.spiderdoc'
    
    :Parameters:
    
    filename : string
              Path of a file
    format : module
              Specific document format
    formats : list
              List of formats, default All
    default_format : module
                    Default format to use if none can be determined
    
    :Returns:
    
    val : module
          Format module
    '''
    
    _logger.debug("get_format_by_ext - start")
    if formats is None: formats=__formats
    if format is None:
        ext = os.path.splitext(filename)[1][1:]
        for f in formats:
            _logger.debug("get_format_by_ext: %s == %s -- %d"%(ext, f.extension(), (ext == f.extension())))
            if ext == f.extension():
                format = f
                break
        if format is None: 
            if default_format is None: raise WriteFormatError, "Cannot find format for "+ext
            else: format = default_format
    return format

def get_format(filename, format=None, formats=None, mode='r', getformat=True, header=None, **extra):
    '''Find appropriate format for a file
    
    This function calls open_file to create the filename, then attempts to parse the header
    with the given formats. If one succeeds, then the format module or the cached header
    and input stream are returned.
    
    .. sourcecode:: py
        
        >>> from arachnid.core.metadata.format import *
        >>> import os
        >>> os.system("more doc001.spi")
        ; ID      X           Y
           1 2  572.00      228.00    
           2 2  738.00      144.00    
           3 2  810.00      298.00
        
        >>> format = get_format("doc001.spi=id,x,y")
        >>> format.extension()
        'dat'
        >>> format.__name__
        'core.metadata.formats.spiderdoc'
    
    :Parameters:
    
    filename : string
              Path of a file
    format : module
              Metadata document format
    formats : list
              List of formats, default All
    mode : string
           File stream read mode
    getformat : bool
             If True return format otherwise return cached header
    header : list
             List of string values describing a file header
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
    
    val1 : module
          Format module (If getformat is true)
    val2 : tuple
          Otherwise, tuple of values including: (input stream, header, factory, last line)
    '''
    
    _logger.debug("get_format: "+str(filename))
    if formats is None: formats=__formats
    filename, fin, header = open_file(filename, header=header, **extra)
    empty_header = (len(header) == 0)
    
    _logger.debug("header: \""+str(header)+"\" -- "+str(len(header))+" -- "+str(header.__class__.__name__))
    if format is not None:
        vals = format.read_header(fin, **extra)
        return  (fin, header) + vals
    for format in formats:
        if empty_header: header = []
        try:
            _logger.debug("trying format: "+str(format.__name__)+" --> "+str(header)+" --> "+str(empty_header)+" --> "+str(len(header)))
            vals = format.read_header(fin, header=header, **extra)
            _logger.debug("Success")
        except:
            if format._logger.isEnabledFor(logging.DEBUG):
                format._logger.exception("Cannot parse with format: "+format.__name__)
            fin = open(filename, mode)
            continue
        return format if getformat else (fin, format) + vals
    raise ParseFormatError, "Cannot find format for "+filename

def read_array_mpi(filename, numeric=True, sort_column=None, **extra):
    ''' Read a file and return as an ndarray (if MPI-enabled, only one process reads
    and the broadcasts to the rest.
    
    :Parameters:
    
    filename : str
               Input filename
    numeric : bool
              Convert each value to float or int (if possible)
    sort_column : int
                  Column to sort the array
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
    
    vals : ndarray
           Table array containing file values
    '''
    
    vals = None
    if mpi_utility.is_root(**extra):
        vals = read(filename, numeric=numeric, **extra)
        vals = format_utility.tuple2numpy(vals)[0]
        if sort_column < vals.shape[1]:
            vals[:] = vals[numpy.argsort(vals[:, sort_column]).squeeze()]
    return mpi_utility.broadcast(vals, **extra)
 
def is_readable(filename, **extra):
    ''' Test if file is readable by a metadata parser
    
    :Parameters:
    
    filename : string
              Path of a file
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
    
    readable : bool
               True if file is readable
    '''
    
    try:
        get_format(filename, getformat=False, **extra)
    except: return False
    return True
    
def read(filename, columns=None, header=None, **extra):
    '''Read a document from the specified file
    
    This function calls open_file to create the filename, then calls get_formats to
    find the appropriate format. It then creates a list of tuples, using the given
    format to parse the file.
    
    .. sourcecode:: py
         
        >>> from arachnid.core.metadata.format import *
        >>> import os
        >>> os.system("more doc001.spi")
        ; ID      X           Y
           1 2  572.00      228.00    
           2 2  738.00      144.00    
           3 2  810.00      298.00
        
        >>> values = read("doc001.spi=id,x,y")
        >>> values
        [BasicTuple(id='1', x='572', y='228'), BasicTuple(id='2', x='738', y='144'), BasicTuple(id='3', x='810', y='298')]
        >>> values[0]._fields
        ('id', 'x', 'y')
        
        >>> os.system("more data.csv")
        id,x,y
        1,572,228
        2,738,144
        3,810,298
        
        >>> values = read("data.csv")
        >>> values
        [BasicTuple(id='1', x='572', y='228'), BasicTuple(id='2', x='738', y='144'), BasicTuple(id='3', x='810', y='298')]
        >>> values._fields
        ('id', 'x', 'y')
    
    :Parameters:
    
    filename : string
              Path of a file
    columns : list, optional
              List of columns to use
    header : str, optional
             Header to use for read-in list
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
    
    val1 : list
           List of namedtuples or other container created by the factory
    '''
    
    _logger.debug("read: "+str(filename))
    fin, format, factory, header, lastline = get_format(filename, getformat=False, header=header, **extra)
    # TODO: columns not finished
    if columns is not None:
        cols = []
        for c in columns:
            try:
                cols.append(int(c))
            except:
                try:
                    cols.append(header.index(c))
                except:
                    raise ValueError, "Cannot find column "+str(c)+" in header: "+",".join(header)
    
    try:
        return map(factory, format.reader(fin, header, lastline, **extra))
    except:
        _logger.debug("header: %s"%str(header))
        raise

def write(filename, values, mode='w', **extra):
    ''' Write a document to some format either specified or determined from extension
    
    .. sourcecode:: py
         
        >>> from arachnid.core.metadata.format import *
        >>> from collections import namedtuple
        >>> BasicTuple = namedtuple("BasicTuple", "id,x,y")
        >>> values = [ BasicTuple("1", 572.00, 228.00 ), BasicTuple("2", 738.00, 144.00 ), BasicTuple("3", 810.00, 298.00) ]
        >>> write("data.dat", values)
        'data.dat'
        >>> import os
        >>> os.system("more data.dat")
        ;            id          x          y
        1  31         572         228
        2  32         738         144
        3  33         810         298

        >>> write("data.csv", values)
        >>> os.system("more data.csv")
        id,x,y
        1,572,228
        2,738,144
        3,810,298
        >>> write("data.ter", values, default_format=csv)
        >>> os.system("more data.ter")
        id,x,y
        1,572,228
        2,738,144
        3,810,298
        >>> write("data.spi", values, format=csv)
        >>> os.system("more data.spi")
        id,x,y
        1,572,228
        2,738,144
        3,810,298
    
    :Parameters:
    
    filename : str
              Input filename containing data
    values : list
             List of objects to be written
    mode : string
           Open file mode, default write over existing
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
    
    val : string
           Path to file created by this function
    '''
    
    if len(values) == 0: raise ValueError, "Nothing to write - array has length 0"
    
    format = get_format_by_ext(filename, **extra)
    filename, fout = open_file(filename, mode=mode, **extra)
    if hasattr(values, 'shape'):
        if 'header' not in extra: raise ValueError, "numpy.ndarray requires that you specify a header"
        values = format_utility.create_namedtuple_list(values, "DefaultFormat", extra['header'])
    else:
        values = format_utility.flatten(values)
        if len(values) > 0 and not hasattr(values[0], '_fields'):
            if 'header' not in extra: raise ValueError, "Cannot find header: %s"%str(values[0].__class__)
            values = format_utility.create_named_list(values, extra['header'], "DefaultFormat")
            
    if mode == 'a': format.write_values(fout, values, **extra)
    else: format.write(fout, values, **extra)
    fout.close()
    return filename

def write_dataset(output, feat, id=None, label=None, good=None, header=None, sort=False, id_len=0, prefix=None, **extra):
    '''Write a set of non-tuple arrays representing a dataset to a file
    
    :Parameters:
        
    output : str
             Name of the output file
    feat : array
           Array of values to be written out
    id : int, optional
         SPIDER id
    label : array, optional
            Array of labels to be the first column
    good : array, optional
           Array of selections to be the second column (if label is not None)
    header : str, optional
             String of comma separated values for the header
    sort : bool, optional
           If True, sort the features and label by the label
    id_len : int, optional
             Maximum length of the SPIDER id
    prefix : str, optional
             Prefix to add to start of the output filename
    extra : dict
            Unused extra keyword arguments
        
    :Returns:
    
    return_val : str
                 Actual output filename used
    '''
    
    if label is not None and sort:
        if label.shape[0] != feat.shape[0]: _logger.error("label does not match size of feat: %d != %d"%(label.shape[0], feat.shape[0]))
        assert(label.shape[0] == feat.shape[0])
        if label.ndim == 2 and label.shape[1] > 1:
            tmp = label.view('i8,i8')
            #tmp = label.view([label.dtype, label.dtype])
            #tmp = label.view([numpy.int, numpy.int])
            idx = numpy.argsort(tmp, axis=0, order=['f0', 'f1']).squeeze()
        elif label.ndim == 1 or (label.ndim == 2 and label.shape[1] == 1):
            idx = numpy.argsort(label, axis=0).squeeze()
        else: raise ValueError, "Sort only support for label of 2 dimensions or less"
        try:
            feat = feat[idx, :].squeeze().copy()
        except:
            _logger.error("%d < %d == %d"%(feat.shape[0], idx.shape[0], label.shape[0]))
            raise
        label = label[idx, :].squeeze().copy()
        if good is not None: good = good[idx, :].squeeze().copy()
    
    mheader = header
    header = ""
    if label is not None:
        if label.ndim == 1 or label.shape[1] < 1: header = "id,"
        elif label.shape[1] < 3:                  header = "id,"
        else:                                     raise ValueError, "Unexpected number of labels"
    if good is not None: header += "select,"
    if mheader is None:
        if hasattr(feat, 'ndim'): cols = feat.shape[1] if feat.ndim > 1 else 1
        else: cols = len(feat[0])
        mheader = ",".join(["c"+str(i) for i in xrange(cols)]) if cols > 1 else "pred"
    else:
        mheader = mheader.split(',')
        if hasattr(feat, 'ndim'): cols = feat.shape[1] if feat.ndim > 1 else 1
        else: cols = len(feat[0])
        cols -= len(mheader)
        mheader = ",".join(mheader+["c"+str(i) for i in xrange(cols)]) if (cols+len(mheader)) > 1 else "pred"
    header = header+mheader
    try:
        vals = format_utility.create_namedtuple_list(feat, "Dataset", header, label, good)
    except:
        _logger.error("header: %s -- len: %d -- shape: %s"%(header, len(label), str(feat.shape)))
        raise
    if id is not None: id = format_utility.fileid(id)
    if 'default_format' not in extra: extra['default_format'] = csv
    return write(output, vals, spiderid=id, id_len=id_len, prefix=prefix, **extra)

def read_identifiers(filename, columns=None, **extra):
    '''Read only the identifier column from the file
    
    This function attempts to parse concatenated identifiers and returns
    them in a list of specially named tuples.
    
    .. sourcecode:: py
         
        >>> from arachnid.core.metadata.format import *
        >>> import os
        >>> os.system("more data.spi")
        ; ID      X           Y
           1 2  572.00      228.00    
           2 2  738.00      144.00    
           3 2  810.00      298.00
        
        >>> read_identifiers("data.spi=id,x,y")
        [1, 2, 3]
        
        >>> os.system("more data.csv")
        id,x,y
        1,572.00,228.00    
        2,738.00,144.00    
        3,810.00,298.00
        
        >>> read_identifiers("data.csv")
        [1, 2, 3]
    
    :Parameters:
    
    filename : string
              Path of a file
    columns : list
             List of objects to be written
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
    
    val : list
          List of identifier tuples or integer ids
    '''
    
    columns=("id",)
    values = read(filename, columns=columns, **extra)
    return split_spider_id(values)

def read_alignment(filename, header=None, **extra):
    ''' Read alignment data from a file
    
    :Parameters:
    
    filename : str
              Input filename containing alignment data
    header : str
             User-specified header for the alignment file
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
    
    align : list
            List of named tuples
    '''
    align_header = [
                    "epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus",
                    "epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,defocus",
                    "epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror"
                ]
    if 'numeric' in extra: del extra['numeric']
    align = None
    for h in align_header:
        try:
            align = read(filename, numeric=True, header=h, **extra)
        except: pass
        else: break
    if align is None:
        align = read(filename, numeric=True, header=header, **extra)
    return align




