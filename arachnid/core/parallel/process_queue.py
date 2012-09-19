''' Utilities to distribute computation in parallel processes 

This module handles parallel process creation and 
utilization with a queue.

.. Created on Oct 16, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import multiprocessing
import logging, sys, traceback, numpy, ctypes, scipy, scipy.sparse
import functools

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def shmem_as_ndarray(raw_array):
    ''' Create a numpy.ndarray view of a multiprocessing.RawArray
        
    :Parameters:
    
    raw_array : multiprocessing.RawArray
                Raw shared memory array
    
    :Returns:
    
    data : numpy.ndarray
           Interface to a RawArray
    
    .. note::
        
        Original code:
        http://pyresample.googlecode.com/svn-history/r2/trunk/pyresample/_multi_proc.py
    
    '''
    _ctypes_to_numpy = {
                        ctypes.c_char : numpy.int8,
                        ctypes.c_wchar : numpy.int16,
                        ctypes.c_byte : numpy.int8,
                        ctypes.c_ubyte : numpy.uint8,
                        ctypes.c_short : numpy.int16,
                        ctypes.c_ushort : numpy.uint16,
                        ctypes.c_int : numpy.int32,
                        ctypes.c_uint : numpy.int32,
                        ctypes.c_long : numpy.int32,
                        ctypes.c_ulong : numpy.int32,
                        ctypes.c_longlong : numpy.longlong,
                        ctypes.c_float : numpy.float32,
                        ctypes.c_double : numpy.float64
                        }
    address = raw_array._wrapper.get_address()
    size = raw_array._wrapper.get_size()
    dtype = _ctypes_to_numpy[raw_array._type_]
    class Dummy(object): pass
    d = Dummy()
    d.__array_interface__ = {
                             'data' : (address, False),
                             'typestr' : numpy.dtype(numpy.uint8).str,
                             'descr' : numpy.dtype(numpy.uint8).descr,
                             'shape' : (size,),
                             'strides' : None,
                             'version' : 3
                             }                            
    return numpy.asarray(d).view(dtype=dtype)

def create_global_dense_matrix(shape, dtype=numpy.float, use_local=False):
    ''' Create a special shared memory array that keeps its shape
    
    :Parameters:
    
    shape : tuple
            Shape of the shared memory array
    dtype : numpy.dtype
            Type of the array
    use_local : bool
                If True create an numpy.ndarray
    
    :Returns:
    
    data : numpy.ndarray
           ndarray view of the shared memory
    shmem : multiprocessing.RawArray
            shared memory array
    '''

    _numpy_to_ctypes = {
                        #numpy.int8: ctypes.c_char,
                        numpy.int16 : ctypes.c_wchar,
                        numpy.int8 : ctypes.c_byte,
                        numpy.uint8 : ctypes.c_ubyte,
                        numpy.int16 : ctypes.c_short,
                        numpy.uint16 : ctypes.c_ushort,
                        numpy.int32 : ctypes.c_int,
                        numpy.int64 : ctypes.c_longlong,
                        numpy.longlong : ctypes.c_longlong,
                        #numpy.int32 : ctypes.c_uint,
                        #numpy.int32 : ctypes.c_long,
                        #numpy.int32 : ctypes.c_ulong,
                        numpy.float32 : ctypes.c_float,
                        numpy.float64 : ctypes.c_double,
                        numpy.float : ctypes.c_double
                        }
    if not isinstance(shape, tuple): shape = (shape, )
    if not use_local:
        tot = 1
        for s in shape: tot *= s
        shmem_data = multiprocessing.RawArray(_numpy_to_ctypes[dtype], tot)
        data = shmem_as_ndarray(shmem_data).reshape(shape)
        shmem = (shmem_data, shape)
    else:
        data = numpy.ndarray(shape, dtype)
        shmem = None
    return data, shmem

def create_global_sparse_matrix(n, neighbors, use_local=False):
    ''' Create a special shared memory array that keeps its shape
    
    :Parameters:
    
    n : int
        Number of rows in the sparse array
    neighbors : int
                Number of columns in the sparse array
    use_local : bool
                If True create an numpy.ndarray
    
    :Returns:
    
    mat : scipy.sparse.coo_matrix
          Sparse matrix with shared memory view
    '''
    
    tot = n*(neighbors+1)
    if not use_local:
        shmem_data = multiprocessing.RawArray(ctypes.c_double, tot)
        shmem_row = multiprocessing.RawArray(ctypes.c_int, tot)
        shmem_col = multiprocessing.RawArray(ctypes.c_int, tot)
        data = shmem_as_ndarray(shmem_data)
        row = shmem_as_ndarray(shmem_row)
        col = shmem_as_ndarray(shmem_col)
        mat = scipy.sparse.coo_matrix( (data,(row, col)), shape=(n, n) )
        mat.shmem = (shmem_data, shmem_row, shmem_col, (n, n))
    else:
        data = numpy.empty(tot, dtype=numpy.float)
        row  = numpy.empty(tot, dtype=numpy.longlong)
        col  = numpy.empty(tot, dtype=numpy.longlong)
        mat = scipy.sparse.coo_matrix( (data,(row, col)), shape=(n, n) )
    return mat

def recreate_global_dense_matrix(shmem):
    ''' Create an ndarray from a shared memory array with the correct shape
    
    :Parameters:
    
    shmem : multiprocessing.RawArray
            shared memory array
    
    :Returns:
    
    data : numpy.ndarray
           ndarray view of the shared memory
    '''
    
    return shmem_as_ndarray(shmem[0]).reshape(shmem[1])

def recreate_global_sparse_matrix(shmem, shape=None):
    ''' Create a scipy.sparse.coo_matrix from a shared memory array with the correct shape
    
    :Parameters:
    
    shmem : multiprocessing.RawArray
            shared memory array
    shape : tuple
            Subset shape of the matrix
    
    :Returns:
    
    mat : scipy.sparse.coo_matrix
          Sparse matrix
    '''
    
    data = shmem_as_ndarray(shmem[0])
    row = shmem_as_ndarray(shmem[1])
    col = shmem_as_ndarray(shmem[2])
    if shape is not None:
        return scipy.sparse.coo_matrix( (data[:shape[1]],(row[:shape[1]], col[:shape[1]])), shape=shape[0] )
    return scipy.sparse.coo_matrix( (data,(row, col)), shape=shmem[3] )

def start_workers_with_output(items, worker_callback, n, init_process=None, **extra):
    '''Start workers and distribute tasks
    
    .. sourcecode:: py
    
        >>> from core.parallel.process_queue import *
        >>> def sum(x, factor=10): return x*factor
        >>> qout = start_workers_with_output(range(1,10), sum, 10)
        >>> for i in xrange(1,10): print qout.get()
        10
        20
        30
        40
        50
        70
        60
        80
        90
    
    :Parameters:

    items : list
            List of items to process in parallel
    worker_callback : function
                      Worker callback function to process an item
    n : int
        Number of processes
    init_process : function
                   Initalize the parameters for the child process
    extra : dict
            Unused keyword arguments
        
    :Returns:
    
    val : Queue
          Output queue
    '''
    
    if n > len(items): n = len(items)
    qin, qout = start_workers(worker_callback, n, init_process, **extra)
    for i in enumerate(items): qin.put(i)
    stop_workers(n, qin)
    qin.close()
    return qout

def start_workers(worker_callback, n, init_process=None, **extra):
    '''Start workers and set the worker callback function
    
    .. sourcecode:: py
    
        >>> from core.parallel.process_queue import *
        >>> def sum(x, factor=10): return x*factor
        >>> qin, qout = start_workers(sum, 10)
        >>> for i in xrange(1,10): qin.put(i)
        >>> for i in xrange(1,10): print qout.get()
        >> stop_workers(10, qin)
        10
        20
        40
        30
        70
        60
        80
        50
        90
    
    :Parameters:

    worker_callback : function
                      Worker callback function to process an item
    n : int
        Number of processes
    init_process : function
                   Initalize the parameters for the child process
    extra : dict
            Unused keyword arguments
        
    :Returns:
    
    val : tuple
          Tuple of Queues (input, output)
    '''
    
    if n == 0: return None, None
    qin = multiprocessing.Queue()
    qout = multiprocessing.Queue()
    for i in xrange(n): 
        target = functools.partial(worker_all, qin=qin, qout=qout, worker_callback=worker_callback, process_number=i, init_process=init_process, **extra)
        p = multiprocessing.Process(target=target)
        p.daemon=True
        p.start()
    return qin, qout

def start_raw_workers(worker_callback, n, *args, **extra):
    '''Start workers and set the worker callback function
    
    .. sourcecode:: py
    
        >>> from core.parallel.process_queue import *
        >>> def sum(x, factor=10): return x*factor
        >>> qin, qout = start_raw_workers(sum, 10)
        >>> for i in xrange(1,10): qin.put(i)
        >>> for i in xrange(1,10): print qout.get()
        >> stop_workers(10, qin)
        10
        20
        40
        30
        70
        60
        80
        50
        90
    
    :Parameters:

    worker_callback : function
                      Worker callback function to process an item
    n : int
        Number of processes
    args : list
            List of additional positional arguments
    extra : dict
            Unused keyword arguments
        
    :Returns:
    
    qin : Queue
          Input queue, send objects to parallel tasks
    qout : Queue
           Output queue, recieve objects from parallel tasks
    '''
    
    if n == 0: return None, None
    qin = multiprocessing.Queue()
    qout = multiprocessing.Queue()
    for i in xrange(n):
        target = functools.partial(worker_callback, **extra)
        multiprocessing.Process(target=target, args=(qin, qout)+args).start()
    return qin, qout

def start_raw_enum_workers(worker_callback, n, total=-1, *args, **extra):
    '''Start workers and set the worker callback function
    
    .. sourcecode:: py
    
        >>> from core.parallel.process_queue import *
        >>> def sum(x, factor=10): return x*factor
        >>> qin, qout = start_raw_enum_workers(sum, 10)
        >>> for i in xrange(1,10): qin.put(i)
        >>> for i in xrange(1,10): print qout.get()
        >> stop_workers(10, qin)
        10
        20
        40
        30
        70
        60
        80
        50
        90
    
    :Parameters:

    worker_callback : function
                      Worker callback function to process an item
    n : int
        Number of processes
    total : int
            Total number of objects allowed in the Queue
    args : list
            List of additional positional arguments
    extra : dict
            Unused keyword arguments
        
    :Returns:
    
    qin : Queue
          Input queue, send objects to parallel tasks
    qout : Queue
           Output queue, recieve objects from parallel tasks
    '''
    
    if n == 0: return None, None
    qin = multiprocessing.Queue(total)
    qout = multiprocessing.Queue(total)
    for i in xrange(n):
        target = functools.partial(worker_callback, **extra)
        multiprocessing.Process(target=target, args=(qin, qout, i, n)+args).start()
    return qin, qout

def stop_workers(n, qin):
    '''Terminate the workers with a signal
    
    This function puts a number of None objects in the input queue
    for each worker process.
    
    .. sourcecode:: py
    
        >>> from core.parallel.process_queue import *
        >>> def sum(x, factor=10): return x*factor
        >>> qin, qout = start_workers(sum, 10)
        >>> for i in xrange(1,10): qin.put(i)
        >>> for i in xrange(1,10): print qout.get()
        >> stop_workers(10, qin)
        10
        20
        40
        30
        70
        60
        80
        50
        90
    
    :Parameters:

    n : int
        Number of processes
    qin : Queue
          Input Queue
    '''
    
    for i in xrange(n): qin.put(None)

def worker_all(qin, qout, worker_callback, init_process=None, **extra):
    '''Runs a generic worker process
    
    This function runs the worker call back in an infinite loop that
    only breaks when a None item is taken from the input queue. If
    an exception is thrown, then all the processes are stopped
    and the exception is placed in the output queue.
    
    :Parameters:

    qin : Queue
          Input Queue
    qout : Queue
           Output Queue
    worker_callback : function
                      Worker callback function to process an item
    init_process : function
                   Initalize the parameters for the child process
    extra : dict
            Unused keyword arguments
    '''

    try:
        if init_process is not None: extra.update(init_process(**extra))
        while True:
            try:
                val = qin.get(True, 5)
            except: continue
            if val is None: 
                if hasattr(qin, "task_done"):  qin.task_done()
                break
            index, val = val
            outval = worker_callback(val, **extra)
            qout.put((index, outval))
            if hasattr(qin, "task_done"): qin.task_done()
    except:
        _logger.exception("Error processing worker")
        qout.put(err_msg())
        if hasattr(qin, "task_done"): qin.task_done()
        while not qin.empty():
            try:
                if qin.get_nowait() is None: 
                    break
                if hasattr(qin, "task_done"): qin.task_done()
            except: pass
    finally:
        qout.put(None)
    _logger.debug("finished")

def err_msg():
    '''Package the exception caught in the process
    
    This function packages the exception caught in the process
    and stores it in a ProcessException.
    
    :Returns:

    val : Exception
          Returns a ProcessException, which stores the original exception
    '''
    
    trace = sys.exc_info()[2]
    try:
        exc_value=str(sys.exc_value)
    except:
        exc_value=''
    
    return ProcessException(str(traceback.format_tb(trace)),str(sys.exc_type),exc_value)

class ProcessException(Exception):
    '''Defines an exception wrapper returned by processes
    
    :Parameters:
    
    trace : string
            Formatted trace back of the exception
    exc_type : string
             Type of the exception
    exc_value : string
                Value of the exception
    '''
    
    def __init__(self, trace=None, exc_type=None, exc_value=None):
        "Create a process exception"
        
        self.trace = trace
        self.exc_type = exc_type
        self.exc_value = exc_value
