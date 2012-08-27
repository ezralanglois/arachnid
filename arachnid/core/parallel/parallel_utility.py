''' Utilities for general parallel code

This module contains general utilities for any type of parallel code.

.. Created on Jul 30, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import numpy

def partition_size(data_size, task_count, dtype=numpy.int, out=None):
    ''' Divide the size of the data into a number of equally sized chucks equal to the number of tasks
    
    :Parameters:

    data_size : int
                Number of data elements
    task_count : int
                 Number of worker tasks
    dtype : numpy.dtype
            Type of the output ndarray
    out : numpy.ndarray
          Output array to hold number of elements for each task
    
    :Returns:
    
    out : numpy.ndarray
          Number of elements for each task
    '''
    
    if out is None: out = numpy.zeros(task_count, dtype=dtype)
    for i in xrange(task_count):
        out[i] = ( (data_size / task_count) + (data_size % task_count > i) )
    return out

def partition_offsets(data_size, task_count, dtype=numpy.int):
    ''' Divide the size of the data into a number of equally sized chucks equal to the number of tasks
    
    :Parameters:

    data_size : int
                Number of data elements
    task_count : int
                 Number of worker tasks
    dtype : numpy.dtype
            Type of the output ndarray
    
    :Returns:
    
    out : numpy.ndarray
          Start element offset for each task
    '''
    
    out = numpy.zeros(task_count+1, dtype=dtype)
    partition_size(data_size, task_count, dtype, out[1:])
    return numpy.cumsum(out, out=out)

def partition_array(data, task_count):
    ''' Divide a list of data elements into a number of equally sized chucks equal to the number of tasks
    
    :Parameters:

    data : list
           Iterable object containing data elements
    task_count : int
                 Number of worker tasks
    
    :Returns:
    
    counts : list
             Subset of data grouped for each task
    '''
    
    data_size = len(data)
    group = []
    k=0
    for i in xrange(task_count):
        size = ( (data_size / task_count) + (data_size % task_count > i) )
        group.append([])
        for j in xrange(size):
            group[-1].append(data[k])
            k+=1
    return group

