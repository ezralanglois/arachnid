''' Generate a table using namedtuple and lists

This module creates a namedtuple from a list of values. It 
dynamically creates a namedtuple class from a list of 
field names. 

A namedtuple is a tuple where the values can be accessed 
either the standard way, integer indicies or using field names.

.. todo:: create record_array version - difficulty is that the type must be known

.. Created on Oct 10, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from collections import namedtuple
from .. import format_utility
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def create(header, first_vals, classname="BasicTuple", **extra):
    '''Create namedtuple from the given header
    
    .. sourcecode:: py
    
        >>> from core.metadata.factories.namedtuple_factory import *
        >>> factory = create(['id', 'peak'])
        >>> factory([10, 0.833])
        BasicTuple(id=10, peak=0.83299999999999996)
    
    :Parameters:
    
    header : list
             List of string values
    first_vals : list
                 List of the initial values
    classname : str
                String name for namedtuple class
                 
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    val : functor
          namedtuple functor that creates namedtuple objects from
          the generated class.
    '''

    return namedtuple(classname, ",".join(header))._make

def get_header(values, header=None, offset=False, **extra):
    '''Get the header from the namedtuple
    
    This function retrieves the header from the named tuple or, if given, returns
    the given header.
    
    .. sourcecode:: py
    
        >>> from core.metadata.factories.namedtuple_factory import *
        >>> TestTuple = namedtuple("TestTuple", "id,select")
        >>> get_header(TestTuple("1", 0))
        ('id', 'select')
        >>> get_header([TestTuple("1", 0)])
        get_header([TestTuple("1", 0)])
        >>> get_header(TestTuple("1", 0), ["id", "class"])
        ['id', 'class']
        >>> get_header(TestTuple("1", 0), offset=True)
        None
    
    :Parameters:
    
    values : object
             List of namedtuples, dictionary of namedtuples or namedtuple
    header : list
             List of string values
    offset : boolean
             If true, return None
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    val : list
          List of string values as header
    '''
    
    #print header
    
    if header is not None: return header
    if offset: return None
    if isinstance(values, dict): return get_header(values[values.keys()[0]], **extra)
    if isinstance(values, list): return get_header(values[0], **extra)
    if hasattr(values, "header"): return values.header()
    return values._fields

def format_iter(values, header, valid_entry=None, **extra):
    ''' Create an iterator that properly formats each row
    
    :Parameters:
    
    values : list
             List of Namedtuples
    header : list
             List of string values
    valid_entry : functor
                  Test if row is valid
    extra : dict
            Unused keyword arguments
    '''
    
    for v in values:
        if valid_entry is not None and not valid_entry(v): continue
        yield get_values(v, header, **extra)

def get_values(value, header=None, float_format="%.8g", **extra):
    '''Get the values from a namedtuple
    
    .. sourcecode:: py
    
        >>> from core.metadata.factories.namedtuple_factory import *
        >>> TestTuple = namedtuple("TestTuple", "id,select")
        >>> get_values(TestTuple("1", 0))
        ['1', '0']
        >>> get_values(values, ["id"])
        ['1']
    
    :Parameters:
    
    value : namedtuple
            Namedtuple with row values
    header : list
             List of string values
    float_format : string
                   String format for numeric values
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    val : list
          List of string value representations of numeric values
    '''
    
    values = []
    if header is None:
        for v in value:
            if isinstance(v, str): values.append(v)
            else: values.append(float_format % float(v))
    else:
        for h in header:
            v = getattr(value, h, 0.0)
            if isinstance(v, str): values.append(v)
            else: 
                try:
                    values.append(float_format % float(v))
                except:
                    _logger.error("Cannot convert: %s for %s"%(str(v), h))
                    raise
    return values

def ensure_container(values, header=None, **extra):
    ''' Ensure ensure input is namedtuple list, if not 
    attempt to create one.
    
    :Parameters:
    
    values : container
             Container holding values
    header : list
             List of string values, optional. Required when
             input is not a container of namedtuples.
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    values : list
             List of namedtuples
    '''
    
    if hasattr(values, 'shape'):
        if header is None: raise ValueError, "numpy.ndarray requires that you specify a header"
        #values = create_namedtuple_list(values, "DefaultFormat", header)
        values = create_named_list(values, header, "DefaultFormat")
    else:
        values = format_utility.flatten(values)
        if len(values) > 0 and not hasattr(values[0], '_fields'):
            if header is None: raise ValueError, "Cannot find header: %s"%str(values[0].__class__)
            values = create_named_list(values, header, "DefaultFormat")
    return values

def create_named_list(values, header, name="SomeList"):
    ''' Create a list of namedtuples from a collection of values
    
    .. sourcecode:: py
    
        >>> from core.metadata.format_utility import *
        >>> values = [ [1,0], [2,1], [3,1] ]
        >>> create_named_list(values, "Selection", "id,select")
        [ Selection(id=1, select=0), Selection(id=2, select=1) Selection(id=3, select=1) ]
    
    :Parameters:
    
    val : collection
          Iterable 2D collection of values
    header : str
             Header of namedtuple
    name : str
            Name of namedtuple class
    
    :Returns:

    val : list
          List of namedtuples
    '''
    
    try:""+header
    except: header=",".join(header)
    Tuple = namedtuple(name, header)
    retvals = []
    index = 0
    for row in values:
        try:
            retvals.append(Tuple._make(row))
        except:
            _logger.error("Row(%d): %d"%(index, len(row)))
            _logger.error("Row: %s"%(str(row)))
            _logger.error("Header: %s"%(str(header)))
            raise
        index += 1
    return retvals



