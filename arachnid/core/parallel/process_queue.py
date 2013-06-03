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

def current_id():
    ''' Get the current process id
    
    :Returns:
    
    id : int
         Current process id
    '''
    
    return multiprocessing.current_process().pid

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
    if isinstance(raw_array, tuple): raw_array = raw_array[0]
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

def create_global_dense_matrix(shape, dtype=numpy.float, shared=True):
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
    
    if 1 == 1:
        import shmarray
        data = shmarray.zeros(shape, dtype=dtype)
        return data, data

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
    if hasattr(dtype, 'type'): dtype = dtype.type
    if shared:
        tot = numpy.prod(shape)
        shmem_data = multiprocessing.RawArray(_numpy_to_ctypes[dtype], tot)
        data = shmem_as_ndarray(shmem_data)
        try:
            data = data.reshape(shape)
        except:
            _logger.error("n=%d, data.shape=%s --- shape: %s"%(tot, str(data.shape), str(shape)))
            raise
        shmem = (shmem_data, shape)
    else:
        data = numpy.empty(shape, dtype)
        shmem = data
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
    if 1 == 1:
        import shmarray
        data = shmarray.zeros(tot)
        row = shmarray.zeros(tot, dtype=numpy.longlong)
        col = shmarray.zeros(tot, dtype=numpy.longlong)
        return scipy.sparse.coo_matrix( (data,(row, col)), shape=(n, n) )
    
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
    
    if hasattr(shmem, 'ndim'): return shmem
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
    
    if hasattr(shmem, 'shape'): return shmem
    data = shmem_as_ndarray(shmem[0])
    row = shmem_as_ndarray(shmem[1])
    col = shmem_as_ndarray(shmem[2])
    if shape is not None:
        return scipy.sparse.coo_matrix( (data[:shape[1]],(row[:shape[1]], col[:shape[1]])), shape=shape[0] )
    return scipy.sparse.coo_matrix( (data,(row, col)), shape=shmem[3] )

def for_mapped(worker_callback, thread_count, size, *args, **extra):
    '''
    '''
    
    def worker_wrap(worker, beg, end, qout, *args2, **kwargs):
        for val in worker(beg, end, *args2, **kwargs):
            qout.put(val)
    
    if thread_count > 1:
        qmax = extra.get('qmax', -1)
        qout = multiprocessing.Queue(qmax)
        counts = numpy.zeros(thread_count, dtype=numpy.int)
        for i in xrange(thread_count):
            counts[i] = ( (size / thread_count) + (size % thread_count > i) )
        offsets = numpy.zeros(counts.shape[0]+1, dtype=numpy.int)
        numpy.cumsum(counts, out=offsets[1:])
        processes = [multiprocessing.Process(target=functools.partial(worker_wrap, process_number=i, **extra), args=(worker_callback, offsets[i], offsets[i+1], qout)+args) for i in xrange(thread_count)]
        for p in processes: p.start()
        #for p in processes: p.join()
        for i in xrange(size):
            yield qout.get()
    else:
        worker_callback(0, size, *args, **extra)

def map_array_out(worker_callback, thread_count, data, *args, **extra):
    '''
    '''
    
    def worker_wrap(worker, offset, qout, *args, **kwargs):
        for i, val in worker(*args, **kwargs):
            qout.put((i+offset, val))
    
    if thread_count > 1:
        qmax = extra.get('qmax', -1)
        qout = multiprocessing.Queue(qmax)
        size = len(data)
        counts = numpy.zeros(thread_count, dtype=numpy.int)
        for i in xrange(thread_count):
            counts[i] = ( (size / thread_count) + (size % thread_count > i) )
        offsets = numpy.zeros(counts.shape[0]+1, dtype=numpy.int)
        numpy.cumsum(counts, out=offsets[1:])
        processes = [multiprocessing.Process(target=functools.partial(worker_callback, process_number=i, **extra), args=(data[offsets[i]:offsets[i+1]], offsets[i], qout)+args) for i in xrange(thread_count)]
        for p in processes: p.start()
        for p in processes: p.join()
        for i in xrange(size):
            yield qout.get()
    else:
        worker_callback(data, *args, **extra)

def start_reduce(worker_callback, thread_count, *args, **extra):
    '''Start workers and set the worker callback function
    
    :Parameters:

    worker_callback : function
                      Worker callback function to process an item
    data : array
           Array to map
    args : list
           Unused positional arguments
    extra : dict
            Unused keyword arguments
    '''
    
    def reduce_worker(qout, *args, **kwargs):
        val = worker_callback(*args, **kwargs)
        qout.put(val)
        if hasattr(qout, "task_done"): qout.task_done()
    
    if thread_count > 1:
        qout = multiprocessing.Queue()
        processes = [multiprocessing.Process(target=functools.partial(reduce_worker, process_number=i, **extra), args=(qout, )+args) for i in xrange(thread_count)]
        for p in processes: p.start()
        for i in xrange(thread_count):
            yield qout.get()
    else:
        yield worker_callback(*args, **extra)

def map_array(worker_callback, thread_count, data, *args, **extra):
    '''Start workers and set the worker callback function
    
    :Parameters:

    worker_callback : function
                      Worker callback function to process an item
    data : array
           Array to map
    args : list
           Unused positional arguments
    extra : dict
            Unused keyword arguments
    '''
    if isinstance(data, int): size = data
    elif isinstance(data, tuple) and len(data) == 2:
        size = len(recreate_global_dense_matrix(data))
    else: size = len(data)
    
    if thread_count > 1:
        counts = numpy.zeros(thread_count, dtype=numpy.int)
        for i in xrange(thread_count):
            counts[i] = ( (size / thread_count) + (size % thread_count > i) )
        offsets = numpy.zeros(counts.shape[0]+1, dtype=numpy.int)
        numpy.cumsum(counts, out=offsets[1:])
        processes = [multiprocessing.Process(target=functools.partial(worker_callback, process_number=i, **extra), args=(offsets[i], offsets[i+1], data)+args) for i in xrange(thread_count)]
        for p in processes: p.start()
        for p in processes: p.join()
    else:
        worker_callback(0, size, data, *args, **extra)
        
def map_reduce_array(worker_callback, thread_count, data, *args, **extra):
    '''Start workers and set the worker callback function
    
    :Parameters:

    worker_callback : function
                      Worker callback function to process an item
    data : array
           Array to map
    args : list
           Unused positional arguments
    extra : dict
            Unused keyword arguments
    '''
    
    def reduce_worker(qout, *args, **kwargs):
        val = worker_callback(*args, **kwargs)
        qout.put(val)
        if hasattr(qout, "task_done"): qout.task_done()
    
    size = len(data)
    if thread_count > 1:
        counts = numpy.zeros(thread_count, dtype=numpy.int)
        for i in xrange(thread_count):
            counts[i] = ( (size / thread_count) + (size % thread_count > i) )
        offsets = numpy.zeros(counts.shape[0]+1, dtype=numpy.int)
        numpy.cumsum(counts, out=offsets[1:])
        qout = multiprocessing.Queue()
        processes = [multiprocessing.Process(target=functools.partial(reduce_worker, process_number=i, **extra), args=(qout, offsets[i], offsets[i+1], data)+args) for i in xrange(thread_count)]
        for p in processes: p.start()
        for i in xrange(thread_count):
            yield qout.get()
    else:
        yield worker_callback(0, size, data, *args, **extra)

def map_reduce_ndarray(worker_callback, thread_count, data, *args, **extra):
    '''Start workers and set the worker callback function
    
    :Parameters:

    worker_callback : function
                      Worker callback function to process an item
    data : array
           Array to map
    args : list
           Unused positional arguments
    extra : dict
            Unused keyword arguments
    '''
    
    def reduce_worker(qout, beg, end, data, *args, **kwargs):
        val = worker_callback(data[beg:end], *args, **kwargs)
        qout.put(val)
        if hasattr(qout, "task_done"): qout.task_done()
    
    size = len(data)
    if thread_count > 1:
        counts = numpy.zeros(thread_count, dtype=numpy.int)
        for i in xrange(thread_count):
            counts[i] = ( (size / thread_count) + (size % thread_count > i) )
        offsets = numpy.zeros(counts.shape[0]+1, dtype=numpy.int)
        numpy.cumsum(counts, out=offsets[1:])
        qout = multiprocessing.Queue()
        processes = [multiprocessing.Process(target=functools.partial(reduce_worker, process_number=i, **extra), args=(qout, offsets[i], offsets[i+1], data)+args) for i in xrange(thread_count)]
        for p in processes: p.start()
        for i in xrange(thread_count):
            yield qout.get()
    else:
        yield worker_callback(data, *args, **extra)

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

def start_raw_enum_workers(worker_callback, n, total=-1, outtotal=-1, *args, **extra):
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
    if outtotal == -1: outtotal = total
    qin = multiprocessing.JoinableQueue(total)
    qout = multiprocessing.Queue(outtotal)
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
