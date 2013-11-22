''' Selection utilities

This module provides a set of functions to handle selections.

.. Created on Nov 4, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import spider_utility, format_utility
import numpy

def select_file_subset(files, select, id_len=0, fill=False):
    ''' Create a list of files based on the given selection
    
    This function serves as an interface for the different
    file naming conventions.
    
    .. seealso:: spider_utility.select_file_subset
    
    :Parameters:
    
    files : list
            List of filenames
    select : array
             Array of file ids
    id_len : int
             Maximum length of SPIDER id
    fill : bool
           Fill missing filenames missing from files with those in the selection file
             
    :Returns:
    
    out : list
          List of selected filenames
    '''
    
    return spider_utility.select_file_subset(files, select, id_len, fill)


def select_subset(vals, select):
    ''' Create a list of objects based on the given selection
    
    >>> from arachnid.core.metadata.selection_utility import *
    >>> select_subset([(1,2), (2,3), (3,4) (5,2), (6,3)], [1,2])
    [(1,2), (2,3)]
    
    >>> select_subset([(1,2), (2,3), (3,4) (5,2), (6,3)], [Select(id=1),Select(id=2)])
    [(1,2), (2,3)]
    
    >>> select_subset([(1,2), (2,3), (3,4) (5,2), (6,3)], [Select(id=1,select=1),Select(id=2,select=1),Select(id=2,select=0)])
    [(1,2), (2,3)]
    
    :Parameters:
    
    vals : list
           List of objects
    select : array
             Array of selected indices where first element is 1 not 0
             
    :Returns:
    
    out : list
          List of selected objects
    '''
    
    if len(select) == 0 or len(vals) == 0: return []
    if hasattr(select[0], 'select'):
        return [vals[s.id-1] for s in select if s.select > 0]
    elif hasattr(select[0], 'id'):
        return [vals[s.id-1] for s in select]
    else:
        return [vals[s[0]-1] for s in select]

def create_selection_doc(n, start=1, micrograph_id=None):
    ''' Create a selection document from a range and optional micrograph id
    
    The default header for the output namedtuple is id,select. If micrograph_id is
    specified, then new header becomes micrograph,particle.
    
    >>> from arachnid.core.metadata.selection_utility import *
    >>> create_selection_doc(3)
    [Selection(id=1, select=1),Selection(id=2, select=1),Selection(id=3, select=1)]
    
    >>> create_selection_doc(3,1,10)
    [Selection(micrograph=10, particle=1),Selection(micrograph=10, particle=2),Selection(micrograph=10, particle=3)]
    
    :Parameters:
    
    n : int
        Length of the range
    start : int
            Starting value for the range, if 0, then all the values are incremented
            by 1.
    micrograph_id : int, optional
                    Micrograph id
                    
    :Returns:
    
    vals : array
           List of namedtuples
    '''
    
    values = numpy.ones((n, 2))
    pid = 0
    header="id,select"
    if micrograph_id is not None:
        pid = 1
        values[:, 0] = micrograph_id
        header = "micrograph,particle"
    values[:, pid] = range(start, start+n)
    if start == 0:  values[:, pid] += 1
    return format_utility.create_namedtuple_list(values, "Selection", header=header)

def create_selection_map(offset, n, micrograph_id):
    ''' Create a selection document that maps a global ID
    to micrograph, stack slice IDs.
    
    >>> from arachnid.core.metadata.selection_utility import *
    >>> create_selection_map(50, 2, 10)
    [Selection(id=50, micrograph=10,slice_id=1),Selection(id=51, micrograph=10,slice_id=2)]
    
    >>> create_selection_map(50, [3,9], 10)
    [Selection(id=50, micrograph=10,slice_id=3),Selection(id=51, micrograph=10,slice_id=9)]

    :Parameters:
    
    offset : int
             Offset for global range of ids
    n : int or array
        List of ids or length (1...n+1)
    micrograph_id : int
                    Micrograph id
                    
    :Returns:
    
    vals : array
           List of namedtuples
    '''
    
    if hasattr(n, '__iter__'):
        ids = numpy.asarray(n)
        n = len(ids)
    else:
        ids = numpy.arange(1, 1+n, dtype=numpy.int)
    values = numpy.ones((n, 3), dtype=numpy.int)
    values[:, 0] = numpy.arange(offset+1, offset+1+n, dtype=numpy.int)
    values[:, 1] = micrograph_id
    values[:, 2] = ids
    header = "id,micrograph,slice_id"
    return format_utility.create_namedtuple_list(values, "Selection", header=header)


