''' Generate a table using namedtuple and lists

This module creates a named tuple from a list of values.

.. Created on Oct 10, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from collections import namedtuple

def create(header, classname="BasicTuple", **extra):
    '''Create namedtuple from the given header
    
    .. sourcecode:: py
    
        >>> from core.metadata.factories.namedtuple_factory import *
        >>> factory = create(['id', 'peak'])
        >>> factory([10, 0.833])
        BasicTuple(id=10, peak=0.83299999999999996)
    
    :Parameters:
    
    header : list
             List of string values
    classname : string
                String name for namedtuple class
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    val : functor
          namedtuple functor
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
            else: values.append(float_format % float(v))
    return values



