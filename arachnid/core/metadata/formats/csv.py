''' Read/Write a table in the comma separated format (CSV)

This module reads from and writes to the comma separated format (CSV), columns are separated by commas and rows by newlines.

An example of the file:

.. container:: bottomnav, topic
    
    | id,select,peak
    | 1/1,1,0.00025182
    | 1/2,1,0.00023578

It supports the following attributes:

    - Extension: csv
    - Filter: CSV (\*.csv)

.. Created on Sep 28, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from .. import format_utility
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def read_header(filename, header=[], **extra):
    '''Parses the header on the first line of the CSV file
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.csv")
        id,select,peak
        1/1,1,0.00025182
        1/2,1,0.00023578
        
        >>> header = []
        >>> fin = open("data.csv", 'r')
        >>> header, first_vals = read_header(fin, header)
        >>> header
        ["id","select","peak"]
        >>> first_vals
        ['1/1','1','0.00025182']
    
    :Parameters:
    
    filename : str or stream
               Input filename or stream
    header : list
             List of strings overriding parsed header
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    header : list
             List of fields describing each column
    first_vals : list
                 List of values from the first data line
    '''
    
    fin = open(filename, 'r') if isinstance(filename, str) else filename
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
            header = line.split(',')
            for key, val in dheader.iteritems():
                header[val] = key
        elif len(header) == 0: header.extend(line.split(','))
        line = fin.readline().strip()
        first_vals = parse_line(line, extra.get('numeric'))
        if line[0] == ';': raise format_utility.ParseFormatError, "Cannot parse Spider file"
        if not isinstance(header, dict) and len(header) != len(first_vals): 
            raise format_utility.ParseFormatError, "Cannot parse header of CSV document - header mismatch - "+str(len(header))+" != "+str(len(first_vals))+" - "+str(header)+" :: "+line
        if isinstance(filename, str): fin.close()
        return header, first_vals
    except:
        fin.close()
        raise
    else:
        fin.close()
    raise format_utility.ParseFormatError, "Cannot parse header of CSV document file - end of document"

def reader(filename, header=[], numeric=False, columns=None, **extra):
    '''Creates a CSV read iterator
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.csv")
        id,select,peak
        1/1,1,0.00025182
        1/2,1,0.00023578
        
        >>> header = []
        >>> fin = open("data.csv", 'r')
        >>> header, first_vals = read_header(fin, header)
        >>> header
        ["id", "select", "peak"]
        >>> factory = namedtuple_factory
        >>> [first_vals]+map(factory, reader(fin, header, numeric=True))
        [ BasicTuple(id="1/1", select=1, peak=0.00025182), BasicTuple(id="1/2", select=1, peak=0.00023578) ]
    
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
          CSV read iterator
    '''
    
    fin = open(filename, 'r') if isinstance(filename, str) else filename
    try:
        for line in fin:
            line = line.strip()
            if line == "": continue
            yield parse_line(line, numeric, columns, len(header))
    finally:
        fin.close()

def parse_line(line, numeric=False, columns=None, hlen=None):
    ''' Parse a line of values in the CSV format
    
        >>> parse_line("filename,0,0,1", True)
        ["filename", 0, 0, 1]
    
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
    
    vals = line.split(",")
    if hlen is not None and hlen != len(vals): 
        raise format_utility.ParseFormatError, "Header length does not match values: "+str(hlen)+" != "+str(len(vals))+" --> "+str(vals)
    if columns is not None: vals = vals[columns]
    if numeric: return [format_utility.convert(v) for v in vals]
    return vals

############################################################################################################
# Write format                                                                                             #
############################################################################################################

def write_header(filename, values, mode, header, csv_separtor=',', **extra):
    '''Write a comma separated value (CSV) header
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select,peak")
        >>> header=BasicTuple._fields
        >>> write_header("data.csv", header)
        
        >>> import os
        >>> os.system("more data.csv")
        id,select,peak
    
    :Parameters:
    
    filename : str or stream
               Output filename or stream
    values : container
             Value container such as a list or an ndarray
    mode : str
           Write mode - if 'a', do not write header
    header : list
             List of strings describing columns in data
    csv_separtor : str
                   Seperator for header values
    extra : dict
            Unused keyword arguments
    '''
    
    if mode != 'a':
        fout = open(filename, 'w') if isinstance(filename, str) else filename
        fout.write(csv_separtor.join(header)+"\n")
        if isinstance(filename, str): fout.close()
    return header

def write_values(filename, values, csv_separtor=',', **extra):
    '''Write comma separated value (CSV) values
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select,peak")
        >>> values = [ BasicTuple("1/1", 1, 0.00025182), BasicTuple("1/2", 1, 0.00023578) ]
        >>> write_values("data.csv", values)
        
        >>> import os
        >>> os.system("more data.csv")
        1/1,1,0.00025182
        1/2,1,0.00023578
    
    :Parameters:
    
    filename : str or stream
               Output filename or stream
    values : container
             Value container such as a list or an ndarray
    csv_separtor : str
                   Seperator for data values
    extra : dict
            Unused keyword arguments
    '''
    
    fout = open(filename, 'w') if isinstance(filename, str) else filename
    for v in values:
        fout.write(csv_separtor.join(v)+"\n")
    if isinstance(filename, str): fout.close()
        
############################################################################################################
# Extension and Filters                                                                                    #
############################################################################################################

def extension():
    '''Get extension of CSV format
    
    :Returns:
    
    val : str
          File extension - csv
    '''
    
    return "csv"

def filter():
    '''Get filter of CSV format
    
    :Returns:
    
    val : str
          File filter - CSV (\*.csv)
    '''
    
    return "CSV (*.csv)"



