''' Read/Write a table in the SPIDER document format (DAT)


This module reads from and writes to the the Spider document format, which describes all the metadata 
required by the single particle reconstruction suite.

An example of the file:

.. container:: bottomnav, topic
    
    | ;tst/klh   04-JUN-2003 AT 13:06:06   doc/tot001.klh
    |    1 2  572.00      228.00    
    |    2 2  738.00      144.00    
    |    3 2  810.00      298.00 

It supports the following attributes:

    - Extension: dat
    - Filter: Spider Doc File (\*.dat)

.. Created on Apr 2, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from .. import format_utility
from ..spider_utility import spider_header_vars
from ..factories import namedtuple_factory
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
#_logger.setLevel(logging.DEBUG)

class read_iterator(object):
    '''Spider document format parsing iterator
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data001.spi")
        ;tst/spi   04-JUN-2003 AT 13:06:06   doc/data001.spi
           1 2  572.00      228.00    
           2 2  738.00      144.00    
           3 2  810.00      298.00 
        
        >>> header = ["id", "x", "y"]
        >>> fin = open("data001.spi", 'r')
        >>> factory, lastline = read_header(fin, header)
        >>> header
        ["id", "x", "y"]
        >>> reader = read_iterator(fin, len(header), lastline)
        >>> map(factory, reader)
        [ BasicTuple("1", 572.00, 228.00 ), BasicTuple("2", 738.00, 144.00 ), BasicTuple("3", 810.00, 298.00) ]
    
    :Parameters:
    
    fin : file stream
          Input file stream
    hlen : integer
           Length of the header
    lastline : string
               Last line read during header parsing, requires CSV parsing now
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
        self.lastline=lastline
        self.numeric=numeric
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
                if line == "" or line[0] == ';': continue
                break
        else:
            line = self.lastline
            self.lastline = ""
        vals = line.split()
        if self.hlen+2 == len(vals):
            if self.numeric: return [format_utility.convert(v) for v in vals[2:]]
            return vals[2:]
        elif self.hlen+1 != len(vals): 
            raise format_utility.ParseFormatError, "Header length does not match values: "+str(self.hlen)+" != "+str(len(vals))+" --> "+str(vals)
        del vals[1]
        if self.columns is not None: vals = vals[self.columns]
        if self.numeric: return [format_utility.convert(v) for v in vals]
        return vals

def read_header(filename, header=[], factory=namedtuple_factory, flags=None, **extra):
    '''Parses the header on the first line of the spider document file
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data001.spi")
        ;tst/spi   04-JUN-2003 AT 13:06:06   doc/data001.spi
           1 2  572.00      228.00    
           2 2  738.00      144.00    
           3 2  810.00      298.00 
        
        >>> header = ["id", "x", "y"]
        >>> fin = open("data001.spi", 'r')
        >>> factory, lastline = read_header(fin, header)
        >>> header
        ["id", "x", "y"]
    
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
    lastline = ""
    try:
        while True:
            line = fin.readline()
            if line == "": break
            line = line.strip()
            if line == "": continue
            if line[0] == ';':
                _logger.debug("Comment: %s"%line)
                lastline=line
                continue
            if lastline != "":
                if isinstance(header, dict):
                    if len(header) == 0: raise ValueError, "Dictionary header cannot have zero elements"
                    dheader = header
                    header = spider_header_vars(lastline)
                    tot = len(line.strip().split())
                    if (len(header)+2) != tot and ( (len(header)+1) != tot or header[0] != "id"):
                        _logger.debug("Creating default header: %s"%str(header))
                        del header[:]
                        header.extend(["column"+str(i+1) for i in xrange(tot-2)])
                    for key, val in dheader.iteritems():
                        header[int(val)] = key
                    _logger.debug("Created spider header from dict: %s"%str(header))
                elif len(header) == 0: 
                    _logger.debug("Extending header: %s"%str(header))
                    header.extend(spider_header_vars(lastline))
                    _logger.debug("Spider header: "+lastline)
            else: raise format_utility.ParseFormatError, "Cannot parser header of spider document - does not start with a ; - \"%s\" - %s"%(lastline[0], lastline)
            tot = len(line.strip().split())
            if (len(header)+2) != tot and ( (len(header)+1) != tot or header[0] != "id"):
                _logger.debug("Adding id: %d > %d -- %s == %s"%((len(header)+2), tot, str(header), line))
                if (len(header)+2) > tot:
                    d = (len(header)+2) - tot
                    del header[d:]
                else:
                    _logger.debug("Creating default header")
                    del header[:]
                    header.extend(["column"+str(i+1) for i in xrange(tot-2)])
                #raise format_utility.ParseFormatError, "Cannot parser header of spider document - header mismatch"+str(header)+" - "+lastline
            if (len(header)+2) == tot and header.count("id") == 0:
                _logger.debug("Insert id at front: %d -- %s"%(header.count("id"), str(header)))
                header.insert(0, "id")
                if flags is not None: flags["index_column"] = True
            else:
                if flags is not None: flags["index_column"] = False
            #if flags is not None: flags["index_column"] = ((len(header)+3) == tot)
            vals = line.split()
            if (float(vals[1])) != len(vals[2:]):
                raise format_utility.ParseFormatError, "Not a valid spider file: %d != %d -> %s != %s"%(int(float(vals[1])), len(vals[2:]), str(vals))
            try:
                _logger.debug("Create header: %s"%str(header))
                return factory.create(header, **extra), header, line
            except:
                del header[:]
                header.extend(["column"+str(i+1) for i in xrange(tot-2)])
                if (len(header)+2) == tot and header.count("id") == 0:
                    header.insert(0, "id")
                _logger.debug("create-default-header: "+str(header))
                return factory.create(header, **extra), header, line
    except:
        fin.close()
        raise
    else:
        fin.close()
    raise format_utility.ParseFormatError, "Cannot parse header of Spider document file - end of document"

def test(filename):
    '''Test if filename supports Spider document format
    
    :Parameters:
    
    filename : str
               Name of file to test
    
    :Returns:
    
    val : bool
          True if file supports Spider document format
    '''
    
    try: read_header(filename)
    except: return False
    return True

def reader(filename, header=[], lastline="", **extra):
    '''Creates a spider document read iterator
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data001.spi")
        ;tst/spi   04-JUN-2003 AT 13:06:06   doc/data001.spi
           1 2  572.00      228.00    
           2 2  738.00      144.00    
           3 2  810.00      298.00 
        
        >>> header = ["id", "x", "y"]
        >>> fin = open("data001.spi", 'r')
        >>> factory, lastline = read_header(fin, header)
        >>> header
        ["id", "x", "y"]
        >>> reader = read_iterator(fin, len(header), lastline)
        >>> map(factory, reader)
        [ BasicTuple("1", 572.00, 228.00 ), BasicTuple("2", 738.00, 144.00 ), BasicTuple("3", 810.00, 298.00) ]
    
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
          spider document read iterator
    '''
    
    fin = open(filename, 'r') if isinstance(filename, str) else filename
    return read_iterator(fin, len(header), lastline, **extra)

############################################################################################################
# Write format                                                                                             #
############################################################################################################

def write(filename, values, factory=namedtuple_factory, **extra):
    '''Write a spider document file
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,x,y")
        >>> values = [ BasicTuple("1", 572.00, 228.00 ), BasicTuple("2", 738.00, 144.00 ), BasicTuple("3", 810.00, 298.00) ]
        >>> write("data.spi", values)
        
        >>> import os
        >>> os.system("more data.spi")
        ; ID      X           Y
           1 2  572.00      228.00    
           2 2  738.00      144.00    
           3 2  810.00      298.00 
    
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

def write_header(fout, values, factory=namedtuple_factory, **extra):
    '''Write a spider document header
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,x,y")
        >>> values = [ BasicTuple("1", 572.00, 228.00 ), BasicTuple("2", 738.00, 144.00 ), BasicTuple("3", 810.00, 298.00) ]
        >>> write_header("data.spi", values)
        
        >>> import os
        >>> os.system("more data.spi")
        ; ID      X           Y
    
    :Parameters:
    
    fout : stream
           Output stream
    values : container
             Value container such as a list or an ndarray
    factory : Factory
              Class or module that creates the container for the values returned by the parser
    extra : dict
            Unused keyword arguments
    '''
    
    header = factory.get_header(values, **extra)
    fout.write(";   ")
    for h in header: fout.write("  "+h.rjust(11))
    fout.write("\n")

def write_values(fout, values, factory=namedtuple_factory, header=None, write_offset=1, **extra):
    '''Write values in the spider document format
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,x,y")
        >>> values = [ BasicTuple("1", 572.00, 228.00 ), BasicTuple("2", 738.00, 144.00 ), BasicTuple("3", 810.00, 298.00) ]
        >>> write_values("data.spi", values)
        
        >>> import os
        >>> os.system("more data.spi")
           1 2  572.00      228.00    
           2 2  738.00      144.00    
           3 2  810.00      298.00 
    
    :Parameters:
    
    fout : stream
           Output stream
    values : container
             Value container such as a list or an ndarray
    factory : Factory
              Class or module that creates the container for the values returned by the parser
    header : list
             List of string describing the header
    write_offset : int, optional
                   ID offset in SPIDER document
    extra : dict
            Unused keyword arguments
    '''
    
    if "float_format" in extra: del extra["float_format"]
    header = factory.get_header(values, header=header, offset=False, **extra)
    index = write_offset
    count = len(header)
    header = factory.get_header(values, header=header, offset=True, **extra)
    for v in values:
        vals = factory.get_values(v, header, float_format="%11g", **extra)
        fout.write("%d %2d " % (index, count))
        fout.write(" ".join(vals))
        fout.write("\n")
        index += 1

############################################################################################################
# Extension and Filters                                                                                    #
############################################################################################################

def extension():
    '''Get extension of spider document format
    
    :Returns:
    
    val : string
          File extension - dat
    '''
    
    return "dat"

def filter():
    '''Get filter of spider document format
    
    :Returns:
    
    val : string
          File filter - Spider Doc File (\*.dat)
    '''
    
    return "Spider Doc File (*.dat)"








