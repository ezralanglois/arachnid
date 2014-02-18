''' Common parallel/serial design patterns

This module defines a set of common tasks that can be performed in parallel or serial.

.. Created on Jun 23, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import process_queue
import logging
import numpy.ctypeslib
import multiprocessing.sharedctypes

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process_mp(process, vals, worker_count, init_process=None, **extra):
    ''' Generator that runs a process functor in parallel (or serial if worker_count 
        is less than 2) over a list of given data values and returns the result
        
    :Parameters:
    
    process : function
              Functor to be run in parallel (or serial if worker_count is less than 2)
    vals : list
           List of items to process in parallel
    worker_count : int
                    Number of processes to run in parallel
    init_process : function
                   Initalize the parameters for the child process
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
        val : object
              Return value of process functor
    '''
    
    #_logger.error("worker_count1=%d"%worker_count)
    if len(vals) < worker_count: worker_count = len(vals)
    #_logger.error("worker_count2=%d"%worker_count)
    
    if worker_count > 1:
        qout = process_queue.start_workers_with_output(vals, process, worker_count, init_process, **extra)
        index = 0
        while index < len(vals):
            val = qout.get()
            if isinstance(val, process_queue.ProcessException):
                index = 0
                while index < worker_count:
                    if qout.get() is None:
                        index += 1;
                raise val
            if val is None: continue
            index += 1
            yield val
    else:
        #_logger.error("worker_count3=%d"%worker_count)
        logging.debug("Running with single process: %d"%len(vals))
        for i, val in enumerate(vals):
            yield i, process(val, **extra)

def iterate_map(for_func, worker, thread_count, queue_limit=None, **extra):
    ''' Iterate over the input value and reduce after finished processing
    '''
    
    if thread_count < 2:
        for val in worker(enumerate(for_func), process_number=0, **extra):
            yield val
        return
    
    
    def queue_iterator(qin, process_number):
        try:
            while True:
                val = qin.get()
                if val is None: break
                yield val
        finally: pass
        #_logger.error("queue-done")
    
    def iterate_map_worker(qin, qout, process_number, process_limit, extra):
        val = None
        try:
            val = worker(queue_iterator(qin, process_number), process_number=process_number, **extra)
        except:
            _logger.exception("Error in child process")
            while True:
                val = qin.get()
                if val is None: break
        finally:
            qout.put(val)
            #qin.get()
    
    if queue_limit is None: queue_limit = thread_count*8
    else: queue_limit *= thread_count
    
    qin, qout = process_queue.start_raw_enum_workers(iterate_map_worker, thread_count, queue_limit, 1, extra)
    try:
        for val in enumerate(for_func):
            qin.put(val)
    except:
        _logger.error("for_func=%s"%str(for_func))
        raise
    for i in xrange(thread_count): qin.put(None)
    #qin.join()
    
    for i in xrange(thread_count):
        val = qout.get()
        #qin.put(None)
        if val is None: raise ValueError, "Exception in child process"
        yield val

def iterate_reduce(for_func, worker, thread_count, queue_limit=None, shmem_array_info=None, **extra):
    ''' Iterate over the input value and reduce after finished processing
    '''
    
    if thread_count < 2:
        yield worker(enumerate(for_func), process_number=0, **extra)
        return
    
    shmem_map=None
    if shmem_array_info is not None:
        shmem_map=[]
        shmem_map_base=[]
        for i in xrange(thread_count):
            base = {}
            arr = {}
            for key in shmem_array_info.iterkeys():
                ar = shmem_array_info[key]
                if ar.dtype.str[1]=='c':
                    typestr = ar.dtype.str[0]+'f'+str(int(ar.dtype.str[2:])/2)
                    ar = ar.view(numpy.dtype(typestr))
                if ar.dtype == numpy.dtype(numpy.float64):
                    typecode="d"
                elif ar.dtype == numpy.dtype(numpy.float32):
                    typecode="f"
                else: raise ValueError, "dtype not supported: %s"%str(ar.dtype)
                
                base[key] = multiprocessing.sharedctypes.RawArray(typecode, ar.ravel().shape[0])
                arr[key] = numpy.ctypeslib.as_array(base[key])
                arr[key] = arr[key].view(shmem_array_info[key].dtype).reshape(shmem_array_info[key].shape)
            shmem_map.append(arr)                                  
            shmem_map_base.append(base)
        del shmem_array_info
    
    def queue_iterator(qin, process_number):
        try:
            while True:
                val = qin.get()
                if val is None: break
                yield val
        finally: pass
        #_logger.error("queue-done")
    #extra['shmem_arr']=shmem_map
    
    def iterate_reduce_worker(qin, qout, process_number, process_limit, extra, shmem_map_base=None):#=shmem_map):
        val = None
        try:
            if shmem_map_base is not None:
                ar = shmem_map_base[process_number]
                ar_map={}
                for key in ar.iterkeys():
                    ar_map[key] = numpy.ctypeslib.as_array(ar[key])
                    ar_map[key] = ar_map[key].view(shmem_map[process_number][key].dtype).reshape(shmem_map[process_number][key].shape)
                extra.update(ar_map)
            val = worker(queue_iterator(qin, process_number), process_number=process_number, **extra)
        except:
            _logger.exception("Error in child process")
            while True:
                val = qin.get()
                if val is None: break
        finally:
            if shmem_map_base is not None:
                qout.put(process_number)
            else:
                qout.put(val)
    
    if queue_limit is None: queue_limit = thread_count*8
    else: queue_limit *= thread_count
    
    qin, qout = process_queue.start_raw_enum_workers(iterate_reduce_worker, thread_count, queue_limit, 1, extra, shmem_map_base)
    try:
        for val in enumerate(for_func):
            qin.put(val)
    except:
        _logger.error("for_func=%s"%str(for_func))
        raise
    for i in xrange(thread_count): qin.put(None)
    #qin.join()
    
    for i in xrange(thread_count):
        val = qout.get()
        if shmem_map is not None:
            val = shmem_map[val]
        #qin.put(None)
        if val is None: raise ValueError, "Exception in child process"
        yield val
        
def for_process_mp(for_func, worker, shape, thread_count=0, queue_limit=None, **extra):
    ''' Generator to process collection of arrays in parallel
    
    :Parameters:
    
    for_func : func
               Generate a list of data
    work : function
           Function to preprocess the images
    thread_count : int
                   Number of threads
    shape : int
            Shape of worker result array
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    index : int
            Yields index of output array
    out : array
          Yields output array of worker
    '''
    
    if thread_count < 2:
        for i, val in enumerate(for_func):
            res = worker(val, i, **extra)
            yield i, res
    else:
        if queue_limit is None: queue_limit = thread_count*8
        else: queue_limit *= thread_count
        qin, qout = process_queue.start_raw_enum_workers(process_worker2, thread_count, queue_limit, -1, worker, extra)
        
        try:
            total = 0
            for i, val in enumerate(for_func):
                if i >= thread_count:
                    pos = qout.get() #if i > thread_count else i
                    if pos is None or pos == -1: raise ValueError, "Error occured in process: %d"%pos
                    res, idx = pos
                    yield idx, res
                else: 
                    pos = i
                    total += 1
                qin.put((val,i))
            for i in xrange(total):
                pos = qout.get()
                if pos is None or pos == -1: raise ValueError, "Error occured in process: %d"%pos
                res, idx = pos
                yield idx, res
        finally:
            #_logger.error("Terminating %d workers"%(thread_count))
            for i in xrange(thread_count): 
                qin.put((-1, -1))
                pos = qout.get()
                if pos != -1:
                    _logger.error("Wrong return value: %s"%str(pos))
                assert(pos==-1)
    raise StopIteration

def process_worker2(qin, qout, process_number, process_limit, worker, extra):
    ''' Worker in each process that preprocesses the images
    
    :Parameters:
    
    qin : multiprocessing.Queue
          Queue with index for input images in shared array
    qout : multiprocessing.Queue
           Queue with index and offset for the output images in shared array
    process_number : int
                     Process number
    process_limit : int
                    Number of processes
    worker : function
             Function to preprocess the images
    shmem_img : multiprocessing.RawArray
                Shared memory image array
    shape : tuple
            Dimensions of the shared memory array
    extra : dict
            Keyword arguments
    '''
    
    _logger.debug("Worker %d of %d - started"%(process_number, process_limit))
    try:
        while True:
            pos = qin.get()
            if pos is None or not hasattr(pos[0], 'ndim'): break
            res, idx = pos
            val = worker(res, idx, **extra)
            qout.put((val, idx))
        _logger.debug("Worker %d of %d - ending ..."%(process_number, process_limit))
        qout.put(-1)
    except:
        _logger.exception("Finished with error")
        qout.put(None)
    else:
        _logger.debug("Worker %d of %d - finished"%(process_number, process_limit))

