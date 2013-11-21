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
from spiderdoc import parse_line, read_header, reader, logging, write_header as write_spider_header, write_values
if parse_line or read_header or reader or write_values: pass # Hack for pyflakes

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

############################################################################################################
# Write format                                                                                             #
############################################################################################################
def write_header(fout, values, mode, header, **extra):
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
    
    return write_spider_header(fout, values, mode, ["id", "select"], **extra)
    
def valid_entry(row): 
    ''' Test if a row in the file is valid
    
    :Parameters:
    
    row : NamedTuple
    
    :Returns:
    
    return_val : bool
                 True if select > 0
    '''
    return row.select > 0




############################################################################################################
# Extension and Filters                                                                                    #
############################################################################################################

def extension():
    '''Get extension of spider document format
    
    :Returns:
    
    val : str
          File extension - sdat
    '''
    
    return "sdat"

def filter():
    '''Get filter of spider document format
    
    :Returns:
    
    val : str
          File filter - Spider Selection Doc (\*.sdat)
    '''
    
    return "Spider Selection Doc (*.sdat)"


