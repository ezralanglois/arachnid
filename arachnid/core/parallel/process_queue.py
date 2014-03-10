''' Utilities to distribute computation in parallel processes 

This module handles parallel process creation and 
utilization with a queue.

.. Created on Oct 16, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import multiprocessing
import logging, sys, traceback, numpy
import functools

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def safe_get(get):
    ''' Ignore EINTR during Queue.get
    
    :Parameters:
    
        get : functor
              Queue.get method
    
    :Returns:
        
        val : object
              Value from Queue
    '''
    
    while True:
        try: return get()
        except IOError, e:
            # Workaround for Python bug
            # http://stackoverflow.com/questions/4952247/interrupted-system-call-with-processing-queue
            if e.errno == errno.EINTR: continue
            else: raise     

def current_id():
    ''' Get the current process id
    
    :Returns:
    
    id : int
         Current process id
    '''
    
    return multiprocessing.current_process().pid

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
            yield safe_get(qout.get)
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
            yield safe_get(qout.get)
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
            yield safe_get(qout.get)
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
            yield safe_get(qout.get)
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
            yield safe_get(qout.get)
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
    
    exc_type, exc_value, trace = sys.exc_info()[:3]
    return ProcessException(str(traceback.format_tb(trace)),str(exc_type),exc_value)

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
