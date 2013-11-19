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
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
#_logger.setLevel(logging.DEBUG)

def read_header(filename, header=[], flags=None, **extra):
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
        >>> factory, first_vals = read_header(fin, header)
        >>> header
        ["id", "x", "y"]
    
    :Parameters:
    
    filename : str or stream
               Input filename or stream
    header : list
             List of strings overriding parsed header
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
            tmp_vals = line.strip().split()
            cnt = int(tmp_vals[1])
            first_vals = parse_line(line, extra.get('numeric'), hlen=len(header))
            if cnt != len(tmp_vals[2:]):
                raise format_utility.ParseFormatError, "Not a valid spider file: %d != %d -> %s"%(cnt, len(tmp_vals[2:]), str(tmp_vals))
            try:
                _logger.debug("Create header: %s"%str(header))
                return header, first_vals
            except:
                del header[:]
                header.extend(["column"+str(i+1) for i in xrange(tot-2)])
                if (len(header)+2) == tot and header.count("id") == 0:
                    header.insert(0, "id")
                _logger.debug("create-default-header: "+str(header))
                return header, first_vals
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

def reader(filename, header=[], numeric=False, columns=None, **extra):
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
        >>> factory, first_vals = read_header(fin, header)
        >>> header
        ["id", "x", "y"]
        >>> [first_vals]+map(factory, reader(fin, header, numeric=True))
        [ BasicTuple(id=1, x=572.00, y=228.00 ), BasicTuple(id=2, x=738.00, y=144.00 ), BasicTuple(id=3, x=810.00, y=298.00) ]
    
    :Parameters:
    
    filename : str or stream
               Input filename or input stream
    header : list
             List of strings overriding parsed header
    numeric : boolean
              If true then convert string values to numeric
    columns : list
              List of columns to read otherwise None (all columns)
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    val : iterator
          spider document read iterator
    '''
    
    fin = open(filename, 'r') if isinstance(filename, str) else filename
    try:
        for line in fin:
            line = line.strip()
            if line == "" or line[0] == ';': continue
            yield parse_line(line, numeric, columns, len(header))
    finally:
        fin.close()

def parse_line(line, numeric=False, columns=None, hlen=None):
    ''' Parse a line of values in the CSV format
    
        >>> parse_line("1 2  572.00      228.00", True)
        [1, 572.00, 228.00]
    
    :Parameters:
    
    line : str
           String to parse
    numeric : boolean
              If true then convert string values to numeric
    columns : list
              List of columns to read otherwise None (all columns)
    hlen : int
           Number of elements in the header, optional
    
    :Returns:
    
    val : list
          List of values parsed from input line
    '''
    
    vals = line.split()
    if hlen is not None and hlen+2 == len(vals):
        if numeric: return [format_utility.convert(v) for v in vals[2:]]
        return vals[2:]
    elif hlen is not None and hlen+1 != len(vals): 
        raise format_utility.ParseFormatError, "Header length does not match values: "+str(hlen)+" != "+str(len(vals))+" --> "+str(vals)
    del vals[1]
    if columns is not None: vals = vals[columns]
    if numeric: return [format_utility.convert(v) for v in vals]
    return vals

############################################################################################################
# Write format                                                                                             #
############################################################################################################

def write_header(fout, values, mode, header, **extra):
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
    mode : str
           Write mode - if 'a', do not write header
    header : list
             List of strings describing columns in data
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    header : list
             List of strings describing columns in data
    '''
    
    if mode != 'a':
        fout.write(";   ")
        for h in header: fout.write("  "+h.rjust(11))
        fout.write("\n")
    return header

def write_values(fout, values, header, write_offset=1, **extra):
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
    header : list
             List of string describing the header
    write_offset : int, optional
                   ID offset in SPIDER document
    extra : dict
            Unused keyword arguments
    '''
    
    index = write_offset
    count = len(header)
    for v in values:
        fout.write("%d %2d " % (index, count))
        fout.write(" ".join(v))
        fout.write("\n")
        index += 1
            
def float_format():
    ''' Format for a floating point number
    
    :Returns:
    
    return_val : str
                 %11g
    '''
    
    return "%11g"

############################################################################################################
# Extension and Filters                                                                                    #
############################################################################################################

def extension():
    '''Get extension of spider document format
    
    :Returns:
    
    val : str
          File extension - dat
    '''
    
    return "dat"

def filter():
    '''Get filter of spider document format
    
    :Returns:
    
    val : str
          File filter - Spider Doc File (\*.dat)
    '''
    
    return "Spider Doc File (*.dat)"








