''' Set of utilities to deal with Python types

This module defines a set of functions to test for Python types:

    - String
    - Float holding an integer
    - String holding an integer
    
TODO:
    
    - Remove is_int and replace name of is_float_int with is_int (is_int seems to be redundant)

.. Created on Sep 29, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

def is_string(obj):
    '''Test if an object is a string
    
    This function attempts to concatenate a Python string to the object
        - if this succeeds, then it is considered a string
        - otherwise, it is not a string
    
    .. sourcecode:: py
    
        >>> from core.metadata.type_utility import *
        >>> is_string('yes')
        True
        >>> is_string(1)
        False
    
    :Parameters:

    obj : string-like object
          A possible string object
        
    :Returns:
        
    return_val : boolean
                 True if object is a string
    '''
    
    try:
        if "+"+obj: pass
        return True
    except:
        return False


def is_float_int(f):
    '''Test if the float value is an integer
    
    This function casts the float to an integer and subtracts it from the float
        - if the result is zero, then return True
        - otherwise, return False
    
    .. sourcecode:: py
    
        >>> from core.metadata.type_utility import *
        >>> is_float_int(1.0)
        True
        >>> is_float_int(1.1)
        False
    
    :Parameters:

    obj : float
          A float value
        
    :Returns:
        
    return_val : boolean
                 True if float holds an integer
    '''
    
    try:
        f = float(f)
        i = int(f)
        if (f-i) == 0: return True
        else: return False
    except:
        return False

def is_int(s):
    '''Test if the string holds an integer
    
    This function attempts to cast the string to a float, then test if the float holds an integer
    
        - if this succeeds, then the string holds an integer
        - otherwise, it is not an integer
    
    .. sourcecode:: py
    
        >>> from core.metadata.type_utility import *
        >>> is_int('1.0')
        True
        >>> is_int('1.1')
        False
    
    :Parameters:

    obj : string
          A string object to test
        
    :Returns:
        
    return_val : boolean
                True if string holds an integer
    '''
    
    try:
        return is_float_int(float(s))
    except: return False


