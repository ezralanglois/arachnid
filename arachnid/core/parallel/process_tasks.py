''' Common parallel/serial design patterns

This module defines a set of common tasks that can be performed in parallel or serial.

.. Created on Jun 23, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import process_queue
import logging, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process_mp(process, vals, worker_count, **extra):
    ''' Generator that runs a process functor in parallel (or serial if worker_count 
        is less than 2) over a list of given data values and returns the result
        
    :Parameters:
    
    process : function
              Functor to be run in parallel (or serial if worker_count is less than 2)
    vals : list
           List of items to process in parallel
    worker_count : int
                    Number of processes to run in parallel
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
        val : object
              Return value of process functor
    '''
    
    if worker_count > 1 and len(vals) > worker_count:
        logging.debug("Running with multiple processes: %d"%worker_count)
        qout = process_queue.start_workers_with_output(vals, process, worker_count, **extra)
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
        logging.debug("Running with single process")
        for i, val in enumerate(vals):
            yield i, process(val, **extra)

def for_process_mp(for_func, worker, shape, thread_count=0, **extra):
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
        length = numpy.prod(shape)
        res, shmem_res = process_queue.create_global_dense_matrix( ( thread_count, length )  )
        qin, qout = process_queue.start_raw_enum_workers(process_worker, thread_count, thread_count, worker, shmem_res, shape, extra)
        
        try:
            total = 0
            for i, val in enumerate(for_func):
                if i > thread_count:
                    pos = qout.get() if i > thread_count else i
                    if pos is None or pos == -1: raise ValueError, "Error occured in process: %d"%pos
                    pos, idx = pos
                    yield idx, res[pos]
                    total += 1
                else: pos = i
                res[pos, :] = val.ravel()
                qin.put((pos,i))
            for i in xrange(i, total):
                pos = qout.get()
                if pos is None or pos == -1: raise ValueError, "Error occured in process: %d"%pos
                pos, idx = pos
                yield idx, res[pos].reshape(shape)
        finally:
            _logger.error("Terminating %d workers"%(thread_count))
            for i in xrange(thread_count): 
                qin.put((-1, -1))
                pos = qout.get()
                assert(pos==-1)
    raise StopIteration

def process_worker(qin, qout, process_number, process_limit, worker, shmem_res, shape, extra):
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
    
    res = process_queue.recreate_global_dense_matrix(shmem_res)
    _logger.debug("Worker %d of %d - started"%(process_number, process_limit))
    try:
        while True:
            pos = qin.get()
            if pos is None or pos[0] == -1: break
            pos, idx = pos
            val = worker(res[pos, :].reshape(shape), idx, **extra)
            res[pos, :val.ravel().shape[0]] = val.ravel()
            qout.put((pos, idx))
        _logger.debug("Worker %d of %d - ending ..."%(process_number, process_limit))
        qout.put(-1)
    except:
        _logger.exception("Finished with error")
        qout.put(None)
    else:
        _logger.debug("Worker %d of %d - finished"%(process_number, process_limit))


