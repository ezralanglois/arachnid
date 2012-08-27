''' Read/Write a table in the STAR format

This module reads from and writes to the STAR format, which is an
alternative to XML.

An example of the file:

.. container:: bottomnav, topic
    
    | data_images
    | loop\_
    | _rlnImageName
    | _rlnDefocusU
    | _rlnDefocusV
    | _rlnDefocusAngle
    | _rlnVoltage
    | _rlnAmplitudeContrast
    | _rlnSphericalAberration
    | 000001@/lmb/home/scheres/data/VP7/all_images.mrcs 13538 13985 109.45 300 0.15 2
    | 000002@/lmb/home/scheres/data/VP7/all_images.mrcs 13293 13796 109.45 300 0.15 2
    | 000003@/lmb/home/scheres/data/VP7/all_images.mcrs 13626 14085 109.45 300 0.15 2

It supports the following attributes:

    - Extension: star
    - Filter: Star (\*.star)

.. Created on Sep 28, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..format_utility import ParseFormatError, convert
from ..factories import namedtuple_factory
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class read_iterator(object):
    '''Start format parsing iterator
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.star")
        data_images
        loop_
        _rlnImageName
        _rlnDefocusU
        _rlnDefocusV
        _rlnDefocusAngle
        _rlnVoltage
        _rlnAmplitudeContrast
        _rlnSphericalAberration
        000001@/lmb/home/scheres/data/VP7/all_images.mrcs 13538 13985 109.45 300 0.15 2
        000002@/lmb/home/scheres/data/VP7/all_images.mrcs 13293 13796 109.45 300 0.15 2
        000003@/lmb/home/scheres/data/VP7/all_images.mcrs 13626 14085 109.45 300 0.15 2
        
        >>> header = []
        >>> fin = open("data.star", 'r')
        >>> factory, lastline = read_header(fin, header)
        >>> header
        ["_rlnImageName","_rlnDefocusU","_rlnDefocusV","_rlnDefocusAngle","_rlnVoltage","_rlnAmplitudeContrast","_rlnSphericalAberration"]
        >>> reader = read_iterator(fin, len(header), lastline)
        >>> map(factory, reader)
        [ BasicTuple("000001@/lmb/home/scheres/data/VP7/all_images.mrcs", 13538, 13985, 109.45, 300, 0.15, 2), BasicTuple("000002@/lmb/home/scheres/data/VP7/all_images.mrcs", 13293, 13796, 109.45, 300, 0.15, 2) ]
    
    :Parameters:
        
    fin : file stream
          Input file stream
    hlen : integer
           Length of the header
    lastline : string
               Last line read during header parsing, requires Star parsing now
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
                if line == "" or line[0] == ';' or line[0] == '#': continue
                break
        else:
            line = self.lastline
            self.lastline = ""
        vals = line.split()
        if self.hlen != len(vals): raise ParseFormatError, "Header length does not match values: "+str(self.hlen)+" != "+str(len(vals))+" --> "+str(vals)
        
        if self.columns is not None: vals = vals[self.columns]
        if self.numeric: return [convert(v) for v in vals]
        return vals

def read_header(filename, header=[], factory=namedtuple_factory, **extra):
    '''Parses the header on the first line of the Star file
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.star")
        data_images
        loop_
        _rlnImageName
        _rlnDefocusU
        _rlnDefocusV
        _rlnDefocusAngle
        _rlnVoltage
        _rlnAmplitudeContrast
        _rlnSphericalAberration
        000001@/lmb/home/scheres/data/VP7/all_images.mrcs 13538 13985 109.45 300 0.15 2
        000002@/lmb/home/scheres/data/VP7/all_images.mrcs 13293 13796 109.45 300 0.15 2
        000003@/lmb/home/scheres/data/VP7/all_images.mcrs 13626 14085 109.45 300 0.15 2
        
        >>> header = []
        >>> fin = open("data.star", 'r')
        >>> factory, lastline = read_header(fin, header)
        >>> header
        ["_rlnImageName","_rlnDefocusU","_rlnDefocusV","_rlnDefocusAngle","_rlnVoltage","_rlnAmplitudeContrast","_rlnSphericalAberration"]
    
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
        while True: # Remove header comments
            line = fin.readline()
            if line == "": raise ParseFormatError, "Not a star file or empty"
            if len(line) >= 5 and line[:5] == "data_": break
        while True:
            line = fin.readline()
            if line == "": raise ParseFormatError, "Not a star file or empty"
            line = line.strip()
            if line != "": break
            
        tmpheader = []
        if line == "loop_":
            _logger.debug("Found loop - header has labels")
            while True:
                line = fin.readline()
                if line == "": raise ParseFormatError, "Unexpected end of header"
                if line[0] != "_": break
                line = line.strip()
                tmpheader.append(line[1:])
            while line[0] == ';' or line[0] == '#':
                line = fin.readline()
                if line == "": raise ParseFormatError, "Unexpected end of file"
                line = line.strip()
            tot = len(line.strip().split())
        else:
            while line[0] == ';' or line[0] == '#':
                line = fin.readline()
                if line == "": raise ParseFormatError, "Unexpected end of file"
                line = line.strip()
            tot = len(line.strip().split())
            tmpheader.extend(["column"+str(i+1) for i in xrange(tot)])
            logging.debug("create-header: "+str(header))
        
        if isinstance(header, dict):
            if len(header) == 0: raise ValueError, "Dictionary header cannot have zero elements"
            for key, val in header.iteritems():
                tmpheader[val] = key
        elif len(header) == 0: header.extend(tmpheader)
        if tot != len(header): raise ParseFormatError, "Header does not match the file: %s"%header
        if isinstance(filename, str): fin.close()
        return factory.create(header, **extra), header, line
    except:
        fin.close()
        raise
    else:
        fin.close()
    raise ParseFormatError, "Cannot parse header of Star document file - end of document"

def reader(filename, header=[], lastline="", **extra):
    '''Creates a Star read iterator
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.star")
        id,select,peak
        1/1,1,0.00025182
        1/2,1,0.00023578
        
        >>> header = []
        >>> fin = open("data.star", 'r')
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
          Star read iterator
    '''
    
    fin = open(filename, 'r') if isinstance(filename, str) else filename
    return read_iterator(fin, len(header), lastline, **extra)

############################################################################################################
# Write format                                                                                             #
############################################################################################################
def write(filename, values, factory=namedtuple_factory, **extra):
    '''Write a metadata (Star) file
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "_rlnImageName,_rlnClassNumber,_rlnDefocusU")
        >>> values = [ BasicTuple("1/1", 1, 0.00025182), BasicTuple("1/2", 1, 0.00023578) ]
        >>> write("data.star", values)
        
        >>> import os
        >>> os.system("more data.star")
        id,select,peak
        1/1,1,0.00025182
        1/2,1,0.00023578
    
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
    
def write_header(filename, values, factory=namedtuple_factory, tag="", blockcode="images", **extra):
    '''Write a comma separated value (Star) header
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select,peak")
        >>> values = [ BasicTuple("1/1", 1, 0.00025182), BasicTuple("1/2", 1, 0.00023578) ]
        >>> write_header("data.star", values)
        
        >>> import os
        >>> os.system("more data.star")
        data_images
        loop_
        _rlnImageName
        _rlnDefocusU
        _rlnDefocusV
        _rlnDefocusAngle
        _rlnVoltage
        _rlnAmplitudeContrast
        _rlnSphericalAberration
    
    :Parameters:
    
    filename : string or stream
               Output filename or stream
    values : container
             Value container such as a list or an ndarray
    factory : Factory
              Class or module that creates the container for the values returned by the parser
    tag : str
          Tag for each header value, e.g. tag=rln
    blockcode : str
                Label for the data block
    extra : dict
            Unused keyword arguments
    '''
    
    fout = open(filename, 'w') if isinstance(filename, str) else filename
    fout.write('data_'+blockcode+'\n')
    fout.write('loop_\n')
    header = factory.get_header(values, **extra)
    for h in header:
        fout.write("_"+tag+h+'\n')
    if isinstance(filename, str): fout.close()
    
def write_values(filename, values, factory=namedtuple_factory, header=None, star_separtor=' ', **extra):
    '''Write comma separated value (Star) values
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select,peak")
        >>> values = [ BasicTuple("1/1", 1, 0.00025182), BasicTuple("1/2", 1, 0.00023578) ]
        >>> write_values("data.star", values)
        
        >>> import os
        >>> os.system("more data.star")
        1/1,1,0.00025182
        1/2,1,0.00023578
    
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
        vals = factory.get_values(v, header, float_format="%11g", **extra)
        fout.write(star_separtor.join(vals)+"\n")
    if isinstance(filename, str): fout.close()
        
############################################################################################################
# Extension and Filters                                                                                    #
############################################################################################################

def extension():
    '''Get extension of Star format
    
    :Returns:
    
    val : string
          File extension - star
    '''
    
    return "star"

def filter():
    '''Get filter of Star format
    
    :Returns:
    
    val : string
          File filter - Star (\*.star)
    '''
    
    return "Star (*.star)"



