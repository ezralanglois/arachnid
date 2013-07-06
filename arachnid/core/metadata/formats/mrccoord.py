''' Read/Write a table in space separated format (LST)

This module reads from and writes to the space separated format (LST), columns are separated by spaces and rows by newlines.

An example of the file:

.. container:: bottomnav, topic
    
    | id select peak
    | 1/1 1 0.00025182
    | 1/2 1 0.00023578

It supports the following attributes:

    - Extension: lst
    - Filter: LST (\*.lst)

.. Created on Jun 28, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from .. import format_utility
from ..factories import namedtuple_factory
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class read_iterator(object):
    '''Space separated value (LST) format parsing iterator
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.lst")
        id select peak
        1/1 1 0.00025182
        1/2 1 0.00023578
        
        >>> header = []
        >>> fin = open("data.lst", 'r')
        >>> factory, lastline = read_header(fin, header)
        >>> header
        ["id", "select", "peak"]
        >>> reader = read_iterator(fin, len(header), lastline)
        >>> map(factory, reader)
        [ BasicTuple("1/1", 1, 0.00025182), BasicTuple("1/2", 1, 0.00023578) ]
    
    :Parameters:
    
    fin : file stream
          Input file stream
    hlen : integer
           Length of the header
    lastline : string
               Last line read during header parsing, requires LST parsing now
    numeric : boolean
              If true then convert string values to numeric
    columns : list
              List of columns to read otherwise None (all columns)
    extra : dict
            Unused keyword arguments
    '''
    
    __slots__=("fin", "hlen", "lastline", "numeric", "columns")
    
    def __init__(self, fin, hlen, lastline="", numeric=False, columns=None, **extra):
        "Create a read iterator"
        
        self.fin = fin
        self.hlen = hlen
        self.lastline = lastline
        self.numeric = numeric
        self.columns = columns
    
    def __iter__(self):
        '''Get iterator for class
        
        This class defines its own iterator.
        
        :Returns:
        
        val : iterator
              Self
        '''
        
        return self
    
    def next(self):
        '''Go to the next non-comment line
        
        This method skips to next non-comment line, parses the line into a list of values
        and returns those values. It raises StopIteration when it is finished.
        
        :Returns:
        
        val : list
              List of values parsed from current line of the file
        '''
        
        if self.lastline == "":
            while True:
                line = self.fin.readline()
                if line == "": 
                    self.fin.close()
                    raise StopIteration
                line = line.strip()
                if line == "": continue
                break
        else:
            line = self.lastline
            self.lastline = ""
        vals = line.split()
        if self.hlen != len(vals): raise format_utility.ParseFormatError, "Header length does not match values: "+str(self.hlen)+" != "+str(len(vals))+" --> "+str(vals)
        
        if self.columns is not None: vals = vals[self.columns]
        if self.numeric: return [format_utility.convert(v) for v in vals]
        return vals

def read_header(filename, header=[], factory=namedtuple_factory, **extra):
    '''Parses the header on the first line of the LST file
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.lst")
        id select peak
        1/1 1 0.00025182
        1/2 1 0.00023578
        
        >>> header = []
        >>> fin = open("data.lst", 'r')
        >>> factory, lastline = read_header(fin, header)
        >>> header
        ["id","select","peak"]
    
    :Parameters:
    
    filename : string or stream
               Input filename or stream
    header : list
             List of strings overriding parsed header
    factory : Factory
              Class or module that creates the container for the values returned by the parser
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    val : container
          Container with the given header values
    '''
    
    fin = open(filename, 'r') if isinstance(filename, str) else filename
    #lastline = ""
    try:
        while True:
            line = fin.readline()
            if line == "": break
            line = line.strip()
            if line != "" and line[0] != '%': break
        _logger.debug("Read in header: %s -- %d"%(line, len(header)))
        if isinstance(header, dict):
            if len(header) == 0: raise ValueError, "Dictionary header cannot have zero elements"
            dheader = header
            header = line.split()
            for key, val in dheader.iteritems():
                header[val] = key
        elif len(header) == 0: header.extend(line.split())
        line = fin.readline().strip()
        if line[0] == ';': raise format_utility.ParseFormatError, "Cannot parse Spider file"
        if not isinstance(header, dict) and len(header) != len(line.split()): raise format_utility.ParseFormatError, "Cannot parse header of LST document - header mismatch - "+str(len(header))+" != "+str(len(line.split()))+" - "+str(header)+" :: "+line
        if isinstance(filename, str): fin.close()
        return factory.create(header, **extra), header, line
    except:
        fin.close()
        raise
    else:
        fin.close()
    raise format_utility.ParseFormatError, "Cannot parse header of LST document file - end of document"

def reader(filename, header=[], lastline="", **extra):
    '''Creates a LST read iterator
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.lst")
        id select peak
        1/1 1 0.00025182
        1/2 1 0.00023578
        
        >>> header = []
        >>> fin = open("data.lst", 'r')
        >>> factory, lastline = read_header(fin, header)
        >>> header
        ["id", "select", "peak"]
        >>> r = reader(fin, header, lastline)
        >>> map(factory, r)
        [ BasicTuple("1/1", 1, 0.00025182), BasicTuple("1/2", 1, 0.00023578) ]
    
    :Parameters:
    
    filename : string or stream
               Input filename or input stream
    header : list
             List of strings overriding parsed header
    lastline : string
              Last line read by header parser, first line to parse
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    val : iterator
          LST read iterator
    '''
    
    fin = open(filename, 'r') if isinstance(filename, str) else filename
    return read_iterator(fin, len(header), lastline, **extra)

############################################################################################################
# Write format                                                                                             #
############################################################################################################

def write(filename, values, factory=namedtuple_factory, **extra):
    '''Write a Space separated value (LST) file
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select,peak")
        >>> values = [ BasicTuple("1/1", 1, 0.00025182), BasicTuple("1/2", 1, 0.00023578) ]
        >>> write("data.lst", values)
        
        >>> import os
        >>> os.system("more data.lst")
        id select peak
        1/1 1 0.00025182
        1/2 1 0.00023578
    
    :Parameters:
    
    filename : string or stream
               Output filename or stream
    values : container
             Value container such as a list or an ndarray
    factory : Factory
              Class or module that creates the container for the values returned by the parser
    extra : dict
            Unused keyword arguments
    '''
    
    fout = open(filename, 'w') if isinstance(filename, str) else filename
    write_header(fout, values, factory, **extra)
    write_values(fout, values, factory, **extra)
    if isinstance(filename, str): fout.close()

def write_header(filename, values, factory=namedtuple_factory, **extra):
    '''Write a Space separated value (LST) header
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select,peak")
        >>> values = [ BasicTuple("1/1", 1, 0.00025182), BasicTuple("1/2", 1, 0.00023578) ]
        >>> write_header("data.lst", values)
        
        >>> import os
        >>> os.system("more data.lst")
        id,select,peak
    
    :Parameters:
    
    filename : string or stream
               Output filename or stream
    values : container
             Value container such as a list or an ndarray
    factory : Factory
              Class or module that creates the container for the values returned by the parser
    extra : dict
            Unused keyword arguments
    '''
    
    fout = open(filename, 'w') if isinstance(filename, str) else filename
    fout.write(" ".join(factory.get_header(values, **extra))+"\n")
    if isinstance(filename, str): fout.close()

def write_values(filename, values, factory=namedtuple_factory, header=None, csv_separtor=' ', **extra):
    '''Write Space separated value (LST) values
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select,peak")
        >>> values = [ BasicTuple("1/1", 1, 0.00025182), BasicTuple("1/2", 1, 0.00023578) ]
        >>> write_values("data.lst", values)
        
        >>> import os
        >>> os.system("more data.lst")
        1/1 1 0.00025182
        1/2 1 0.00023578
    
    :Parameters:
    
    filename : string or stream
               Output filename or stream
    values : container
             Value container such as a list or an ndarray
    factory : Factory
              Class or module that creates the container for the values returned by the parser
    extra : dict
            Unused keyword arguments
    '''
    
    fout = open(filename, 'w') if isinstance(filename, str) else filename
    header = factory.get_header(values, offset=True, header=header, **extra)
    for v in values:
        vals = factory.get_values(v, header, **extra)
        fout.write(csv_separtor.join(vals)+"\n")
    if isinstance(filename, str): fout.close()
        
############################################################################################################
# Extension and Filters                                                                                    #
############################################################################################################

def extension():
    '''Get extension of LST format
    
    :Returns:
    
    val : string
          File extension - lst
    '''
    
    return "lst"

def filter():
    '''Get filter of LST format
    
    :Returns:
    
    val : string
          File filter - LST (\*.lst)
    '''
    
    return "LST (*.lst)"



