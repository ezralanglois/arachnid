''' Selection utilities

This module provides a set of functions to handle selections.

.. Created on Nov 4, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import spider_utility

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



