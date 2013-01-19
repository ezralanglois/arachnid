'''Read/Write a table in the SPIDER selection format (SDAT)

This module reads from and writes to the selection format (SDAT), which 
describes the limited, yet important subset of the spider document file.

An example of the file:

.. container:: bottomnav, topic
    
    | ;spl/ter   28-SEP-2009 AT 19:04:44   particles_00001.ter
    | ; /     file_number       class
    | 1  2           1           1 
    | 2  2           2           1

It supports the following attributes:

    - Extension: sdat
    - Filter: Spider Selection Doc (\*.sdat)

.. Created on Apr 9, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from spiderdoc import read_iterator, read_header, reader, logging, namedtuple_factory, write_header as write_spider_header
if read_iterator or read_header or reader: pass # Hack for pyflakes

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

############################################################################################################
# Write format                                                                                             #
############################################################################################################
def write_header(fout, values, factory=namedtuple_factory, **extra):
    '''Write a spider selection header
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select")
        >>> values = [ BasicTuple("1", 1 ), BasicTuple("2", 1 ) ]
        >>> write_header("data.spi", values)
        
        >>> import os
        >>> os.system("more data.spi")
        ;         ID        SELECT
    
    This function forces the header to be: id,select
    
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
    
    extra["header"] = ["id", "select"]
    write_spider_header(fout, values, factory, **extra)
    
def write_values(fout, values, factory=namedtuple_factory, header=None, **extra):
    '''Write values in the spider selection format
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select")
        >>> values = [ BasicTuple("1", 1 ), BasicTuple("2", 1 ) ]
        >>> write_values("data.spi", values)
        
        >>> import os
        >>> os.system("more data.spi")
           1 2     1          1  
           2 2     2          1
    
    :Parameters:
    
    fout : stream
           Output stream
    values : container
             Value container such as a list or an ndarray
    factory : Factory
              Class or module that creates the container for the values returned by the parser
    header : list
             List of string describing the header
    extra : dict
            Unused keyword arguments
    '''
    
    header = ["id", "select"]
    if "float_format" in extra: del extra["float_format"]
    header = factory.get_header(values, offset=False, header=header, **extra)
    
    try:
        if len(values) > 0:
            sel = list(values[0]._fields).index("select")
        else: sel = -1
    except:
        logging.error("select not found in list: "+str(header))
        raise
    
    index = 1
    count = len(header)
    header = factory.get_header(values, header=header, offset=True, **extra)
    
    print len(values)
    for v in values:
        #if v.select > 0:
        print v, "index:", sel
        if int(float(v[sel])) > 0:
            vals = factory.get_values(v, header, float_format="%11g", **extra)
            fout.write("%d %2d " % (index, count))
            fout.write(" ".join(vals))
            fout.write("\n")
            #fout.write("%d %2d %11g %11g\n" % (index, 2, vals.id, vals.select))
            index += 1

def write(filename, values, factory=namedtuple_factory, **extra):
    '''Write a spider document file
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select")
        >>> values = [ BasicTuple("1", 1 ), BasicTuple("2", 1 ) ]
        >>> write("data.spi", values)
        
        >>> import os
        >>> os.system("more data.spi")
        ;         ID        SELECT
           1 2     1          1  
           2 2     2          1
    
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


############################################################################################################
# Extension and Filters                                                                                    #
############################################################################################################

def extension():
    '''Get extension of spider document format
    
    :Returns:
    
    val : string
          File extension - sdat
    '''
    
    return "sdat"

def filter():
    '''Get filter of spider document format
    
    :Returns:
    
    val : string
          File filter - Spider Selection Doc (\*.sdat)
    '''
    
    return "Spider Selection Doc (*.sdat)"


