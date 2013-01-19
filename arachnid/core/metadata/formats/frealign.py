''' Read/Write a table in the Frealign PAR format

This module reads from and writes to the Frealign PAR format.

An example of the file:

.. container:: bottomnav, topic
    
|    C  Align-Fspace parameter file                                                  
|    C                                                                               
|    C           PSI   THETA     PHI     SHX     SHY    MAG   FILM      DF1      DF2 
|          1   8.256 120.864 223.096  -2.512   2.705  50000.  8743  31776.1  31280.3    0.82   0.22
|          2 190.212  77.496  85.677  -1.536   6.947  50000.  8743  31776.1  31280.3    0.82   0.24
|          3 284.119  88.178 224.544   6.250  -1.101  50000.  8743  31776.1  31280.3    0.82   0.28
|          4 190.580  92.432  42.037   1.534   0.438  50000.  8743  31776.1  31280.3    0.82   0.31
|          5 275.692 269.766 288.658   1.054   2.737  50000.  8743  31776.1  31280.3    0.82   0.32
|          6 120.020 129.086 295.909   8.228  -2.400  50000.  8743  31776.1  31280.3    0.82   0.31
|          7 282.421  80.952 176.528   3.590   2.651  50000.  8743  31776.1  31280.3    0.82   0.31
|          8  63.624 295.406 120.192   6.332  -5.979  50000.  8743  31776.1  31280.3    0.82   0.29

It supports the following attributes:

    - Extension: par
    - Filter: Frealign (\*.par)

.. Created on Sep 28, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from .. import format_utility
from ..factories import namedtuple_factory
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class read_iterator(object):
    '''Start format parsing iterator
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.par")
        C  Align-Fspace parameter file                                                  
        C                                                                               
        C           PSI   THETA     PHI     SHX     SHY    MAG   FILM      DF1      DF2 
              1   8.256 120.864 223.096  -2.512   2.705  50000.  8743  31776.1  31280.3    0.82   0.22
              2 190.212  77.496  85.677  -1.536   6.947  50000.  8743  31776.1  31280.3    0.82   0.24
              3 284.119  88.178 224.544   6.250  -1.101  50000.  8743  31776.1  31280.3    0.82   0.28
              4 190.580  92.432  42.037   1.534   0.438  50000.  8743  31776.1  31280.3    0.82   0.31
              5 275.692 269.766 288.658   1.054   2.737  50000.  8743  31776.1  31280.3    0.82   0.32
              6 120.020 129.086 295.909   8.228  -2.400  50000.  8743  31776.1  31280.3    0.82   0.31
              7 282.421  80.952 176.528   3.590   2.651  50000.  8743  31776.1  31280.3    0.82   0.31
              8  63.624 295.406 120.192   6.332  -5.979  50000.  8743  31776.1  31280.3    0.82   0.29
        
        >>> header = []
        >>> fin = open("data.par", 'r')
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
        if self.hlen != len(vals):
            try:val1, val2 = vals[-1].split('-')
            except:
                try:val1, val2, val3 = vals[-1].split('-')
                except: pass
                else:
                    val1+=val2
                    vals[-1]=val1
                    vals.append(val3)
            else:
                vals[-1]=val1
                vals.append(val2)
        if self.hlen != len(vals): raise format_utility.ParseFormatError, "Header length does not match values: "+str(self.hlen)+" != "+str(len(vals))+" --> "+str(vals)
        
        if self.columns is not None: vals = vals[self.columns]
        if self.numeric: return [format_utility.convert(v) for v in vals]
        return vals

def read_header(filename, header=[], factory=namedtuple_factory, **extra):
    '''Parses the header on the first line of the Star file
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.par")
        
C  Align-Fspace parameter file                                                  
C                                                                               
C           PSI   THETA     PHI     SHX     SHY    MAG   FILM      DF1      DF2 
      1   8.256 120.864 223.096  -2.512   2.705  50000.  8743  31776.1  31280.3    0.82   0.22
      2 190.212  77.496  85.677  -1.536   6.947  50000.  8743  31776.1  31280.3    0.82   0.24
      3 284.119  88.178 224.544   6.250  -1.101  50000.  8743  31776.1  31280.3    0.82   0.28
      4 190.580  92.432  42.037   1.534   0.438  50000.  8743  31776.1  31280.3    0.82   0.31
      5 275.692 269.766 288.658   1.054   2.737  50000.  8743  31776.1  31280.3    0.82   0.32
      6 120.020 129.086 295.909   8.228  -2.400  50000.  8743  31776.1  31280.3    0.82   0.31
      7 282.421  80.952 176.528   3.590   2.651  50000.  8743  31776.1  31280.3    0.82   0.31
      8  63.624 295.406 120.192   6.332  -5.979  50000.  8743  31776.1  31280.3    0.82   0.29
        
        >>> header = []
        >>> fin = open("data.par", 'r')
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
            if line == "": raise format_utility.ParseFormatError, "Not a Frealign file or empty"
            if line[0] != 'C': break
        
        vals = line.split()
        try:id = int(vals[0])
        except: id=-1
        if len(vals) < 2 or id != 1: raise format_utility.ParseFormatError, "Not a valid PAR file"
        tmpheader = ['id', 'psi', 'theta', 'phi', 'shx', 'shy', 'mag', 'film', 'defocusu', 'defocusv', 'unk1', 'unk2']
        #PSI   THETA     PHI     SHX     SHY    MAG   FILM      DF1      DF2 
        tot = len(vals)
        
        
        if isinstance(header, dict):
            if len(header) == 0: raise ValueError, "Dictionary header cannot have zero elements"
            for key, val in header.iteritems():
                tmpheader[val] = key
        elif len(header) == 0: header.extend(tmpheader)
        if tot != len(header): raise format_utility.ParseFormatError, "Header does not match the file: %s"%header
        if isinstance(filename, str): fin.close()
        return factory.create(header, **extra), header, line
    except:
        fin.close()
        raise
    else:
        fin.close()
    raise format_utility.ParseFormatError, "Cannot parse header of Star document file - end of document"

def reader(filename, header=[], lastline="", **extra):
    '''Creates a Star read iterator
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.par")
        id,select,peak
        1/1,1,0.00025182
        1/2,1,0.00023578
        
        >>> header = []
        >>> fin = open("data.par", 'r')
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
    '''Write a metadata (Frealign) file
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "_rlnImageName,_rlnClassNumber,_rlnDefocusU")
        >>> values = [ BasicTuple("1/1", 1, 0.00025182), BasicTuple("1/2", 1, 0.00023578) ]
        >>> write("data.par", values)
        
        >>> import os
        >>> os.system("more data.par")
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
    
    raise ValueError, "Not implemented"
    
def write_header(filename, values, factory=namedtuple_factory, tag="", blockcode="images", **extra):
    '''Write a comma separated value (Star) header
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select,peak")
        >>> values = [ BasicTuple("1/1", 1, 0.00025182), BasicTuple("1/2", 1, 0.00023578) ]
        >>> write_header("data.par", values)
        
        >>> import os
        >>> os.system("more data.par")
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
    
    raise ValueError, "Not implemented"
    
def write_values(filename, values, factory=namedtuple_factory, header=None, star_separtor=' ', **extra):
    '''Write comma separated value (Star) values
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select,peak")
        >>> values = [ BasicTuple("1/1", 1, 0.00025182), BasicTuple("1/2", 1, 0.00023578) ]
        >>> write_values("data.par", values)
        
        >>> import os
        >>> os.system("more data.par")
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
    
    raise ValueError, "Not implemented"
        
############################################################################################################
# Extension and Filters                                                                                    #
############################################################################################################

def extension():
    '''Get extension of Star format
    
    :Returns:
    
    val : string
          File extension - par
    '''
    
    return "par"

def filter():
    '''Get filter of Star format
    
    :Returns:
    
    val : string
          File filter - Frealign (\*.par)
    '''
    
    return "Frealign (*.par)"



