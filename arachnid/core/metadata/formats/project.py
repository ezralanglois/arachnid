''' Read/Write a table in the project format (PRJ)

This module reads from and writes to the project format, which is the same as the CSV format.

It supports the following attributes:

    - Extension: prj
    - Filter: Project (\*.prj)


.. Created on Apr 2, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from csv import read_iterator, read_header, reader, logging, write_header, write_values, write
if read_iterator or read_header or reader or write_header or write_values or write: pass # Hack for pyflakes
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def extension():
    '''Get extension of project format
    
    :Returns:
    
    val : string
          File extension - prj
    '''
    
    return "prj"

def filter():
    '''Get filter of project format
    
    :Returns:
    
    val : string
          File filter - Project (\*.prj)
    '''
    
    return "Project (*.prj)"

def has_extension(filename):
    '''Test if filename has correct extension for format
    
    :Parameters:
    
    filename : str
              Filename to test
                  
    :Returns:
    
    val : bool
          True if extension matchs format extension
    '''
    
    return os.path.splitext(filename)[1][1:] == extension()





