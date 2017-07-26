''' MPI utility functions

.. Created on Jul 17, 2011
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy, logging
import parallel_utility
import process_tasks
import socket, os
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
try:
    MPI=None
    from mpi4py import MPI #@UnresolvedImport
except:
    from ..app import tracing
    tracing.log_import_error('mpi4py failed to load', _logger)
    #_logger.addHandler(logging.StreamHandler())
    #logging.exception("mpi4py failed to load")
    #logging.warn("MPI not loaded, please install mpi4py")

	
def mpi_dtype(dtype):
	''' Get the MPI type from the NumPy dtype
	'''
	
	return MPI.__TypeDict__[dtype.char] if hasattr(MPI, '__TypeDict__') else MPI._typedict[dtype.char]

def hostname():
    ''' Get the current hostname
    
    :Return:
    
    name : str
           Hostname of current node
    '''
    
    return socket.gethostname()

def gather_array(vals, curvals=None, comm=None, **extra):
    ''' Gather a distributed array to the root node
    
    :Parameters:
        
    vals : int
            Total number of value to process
    curvals : int
              Rank of the sender
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    extra : dict
            Unused keyword arguments
    '''
    
    if comm is not None:
        comm.barrier()
        _logger.debug("Start gather for node %d - %s"%(comm.Get_rank(), socket.gethostname()))
        if 1 == 1:
            size = comm.Get_size()
            rank = comm.Get_rank()
            if curvals is None:
                b, e = mpi_range(len(vals), rank, comm)
                curvals = vals[b:e]
            if comm.Get_rank() > 0:
                comm.Send([curvals, MPI.DOUBLE], dest=0, tag=1)
            else:
                b, e = mpi_range(len(vals), rank, comm)
                vals[b:e] = curvals
                for node in xrange(1, size):
                    b, e = mpi_range(len(vals), node, comm)
                    comm.Recv([vals[b:e], MPI.DOUBLE], source=node, tag=1)
        else:
            size = comm.Get_size()
            
            '''
            counts = data_range(len(vals), size, vals.shape[1])
            if curvals is None:
                b, e = mpi_range(len(vals), comm.Get_rank(), comm)
                curvals = vals[b:e]
            curvals = curvals.ravel()
            vals = vals.ravel()
            assert(len(curvals) == counts[comm.Get_rank()])
            assert(numpy.sum(counts) == len(vals))
            
            #comm.Allgatherv(sendbuf=[curvals, MPI.DOUBLE], recvbuf=[vals, (counts, None), MPI.DOUBLE])
            _logger.debug("Gather-started: %s"%str(counts))
            comm.Gatherv(sendbuf=[curvals, MPI.DOUBLE], recvbuf=[vals, (counts, None), MPI.DOUBLE])
            _logger.debug("Gather-finished")
            '''
 
def gather_all(vals, curvals=None, comm=None, **extra):
    '''
    '''
    
    if comm is not None:
        comm.barrier()
        counts = parallel_utility.partition_size(len(vals), comm.Get_size())*vals.shape[1]
        if curvals is None:
            b, e = mpi_range(len(vals), comm.Get_rank(), comm)
            curvals = vals[b:e]
        curvals = curvals.ravel()
        vals = vals.ravel()
		mpi_type = mpi_dtype(vals.dtype)
        comm.Allgatherv(sendbuf=[curvals, mpi_type], recvbuf=[vals, (counts, None), mpi_type])
    
def mpi_range(total, rank=None, comm=None, **extra):
    '''Range of values to process for the current node
    
    :Parameters:
        
    total : int
            Total number of value to process
    root : int
           Rank of the sender
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    beg : int
          Start of range to process
    end : int
          End of range to process
    '''
    
    if comm is None: return 0, total
    size = comm.Get_size()
    if rank is None: rank = comm.Get_rank()
    offsets = numpy.cumsum(parallel_utility.partition_size(total, size))
    if rank == 0: return 0, offsets[rank]
    return offsets[rank-1], offsets[rank]

def mpi_slice(total, rank=None, comm=None, **extra):
    '''Range of values to process for the current node
    
    :Parameters:
        
    total : int
            Total number of value to process
    root : int
           Rank of the sender
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    slice : slice
            Valid range to process
    '''
    
    beg, end = mpi_range(total, rank, comm)
    return slice(beg,end)


def iterate_reduce(data, comm=None, **extra):
    '''
    '''
    
    if comm is not None:
        mpi_type = mpi_dtype(data.dtype)
        if comm.Get_rank() == 0:
            if isinstance(data, tuple):
                data = [d.copy() for d in data]
                for node in xrange(1, comm.Get_size()):
                    for d in data:
                        comm.Recv([d, mpi_type], source=node, tag=4)
                    yield data
            else:
                data = data.copy()
                for node in xrange(1, comm.Get_size()):
                    comm.Recv([data, mpi_type], source=node, tag=4)
                    yield data
        else:
            if isinstance(data, tuple):
                for d in data:
                    comm.Send([d, mpi_type], dest=0, tag=4)
            else:
                comm.Send([data, mpi_type], dest=0, tag=4)
    raise StopIteration

def send_to_root(data, root, comm=None, **extra):
    ''' Send specified data array to the root node
    
    :Parameters:
    
    data : array
           Array of data to send to the root (or if root, receive)
    root : int
           Rank of the sender
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    extra : dict
            Unused keyword arguments
    '''
    
    if comm is None or data is None: return
    mpi_type = mpi_dtype(data.dtype)
    rank = comm.Get_rank()
    if rank == 0:
        comm.Recv([data, mpi_type], source=root, tag=4)
    elif rank == root:
        comm.Send([data, mpi_type], dest=0, tag=4)

def broadcast(data, comm=None, **extra):
    ''' Broadcast the specified data to all the nodes
    
    :Parameters:
    
    data : object
           Python object to broadcast to all nodes
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    data : object
           Python object recieved
    '''
    
    if comm is not None:
        data = comm.bcast(data)
    return data

def block_reduce(data, comm=None, batch_size=100000, **extra):
    ''' Reduce data array to the root node
    
    :Parameters:
    
    data : array
           Array of data to send to the root (or if root, receive)
    batch_size : int
                 Total data to reduce at one time
    root : int
           Rank of the root node
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    extra : dict
            Unused keyword arguments
    '''
    
    if comm is None: return
    mpi_type = mpi_dtype(data.dtype)
    if 1 == 0:
        tmp = data.copy(order=data.order)
        #comm.Allreduce(MPI.IN_PLACE, [data, mpi_type], op=MPI.SUM)
        comm.Allreduce([data, mpi_type], [tmp, mpi_type], op=MPI.SUM)
        data[:]=tmp
    else:
        #tmp = numpy.empty(batch_size, dtype=data.dtype)
        batch_count = (data.shape[0]-1) / batch_size + 1
        block_end = 0
        for batch in xrange(batch_count):
            #_logger.error("reduce: %d of %d"%(batch, batch_count))
            block_beg = block_end
            block_end = min(block_beg+batch_size, data.shape[0])
            #comm.Reduce(MPI.IN_PLACE, [data[block_beg:block_end], MPI.FLOAT], op=MPI.SUM, root=0)
            #comm.Allreduce([data[block_beg:block_end], mpi_type], [tmp, mpi_type], op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, [data[block_beg:block_end], mpi_type], op=MPI.SUM)
            #data[block_beg:block_end]=tmp
            comm.barrier()
            
        '''
        batch_count = (data.shape[0]-1) / batch_size + 1
        block_end = 0
        rank = comm.Get_rank()
        for batch in xrange(batch_count):
            block_beg = block_end
            block_end = min(block_beg+batch_size, data.shape[0])
            if rank == root:
                comm.Reduce(MPI.IN_PLACE, [data[block_beg:block_end], MPI.FLOAT], op=MPI.SUM, root=root)
        '''
    return data

def block_reduce_root(data, batch_size=100000, root=0, comm=None, **extra):
    ''' Reduce data array to the root node
    
    :Parameters:
    
    data : array
           Array of data to send to the root (or if root, receive)
    batch_size : int
                 Total data to reduce at one time
    root : int
           Rank of the root node
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    extra : dict
            Unused keyword arguments
    '''
    
    mpi_type = mpi_dtype(data.dtype)
    if comm is None: return
    batch_count = (data.shape[0]-1) / batch_size + 1
    block_end = 0
    rank = comm.Get_rank()
    for batch in xrange(batch_count):
        block_beg = block_end
        block_end = min(block_beg+batch_size, data.shape[0])
        if rank == 0:
            comm.Reduce(MPI.IN_PLACE, [data[block_beg:block_end], mpi_type], op=MPI.SUM, root=root)
        elif rank == root:
            comm.Reduce([data[block_beg:block_end], mpi_type], None, op=MPI.SUM, root=root)
        comm.barrier()

def mpi_init(params, use_MPI=False, **extra):
    ''' Setup the parameters for MPI, if enabled
    
    :Parameters:
    
    params : dict
            Global parameters to setup
    use_MPI : bool
              User-specified parameter for MPI
    extra : dict
            Unused keyword arguments
    '''
    
    if use_MPI and supports_MPI():
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        h = logging.StreamHandler()
        _logger.addHandler(h)
        _logger.info("MPI initialized on node(%d): %s"%(rank, hostname()))
        _logger.removeHandler(h)
        MPI.COMM_SELF.Set_errhandler(MPI.ERRORS_ARE_FATAL)
        MPI.COMM_WORLD.Set_errhandler(MPI.ERRORS_ARE_FATAL)
        params['comm'] = comm
        params['rank'] = rank
    elif use_MPI:
        raise ValueError, "MPI failed to initlize - please install mpi4py"
    else:
        params['rank'] = 0

def supports_MPI():
    ''' Test if mpi4py can be imported
    
    :Returns:
    
    val : bool
          True if mpi4py can be imported
    '''
    
    return MPI is not None

def mpi_reduce(process, vals, comm=None, rank=None, **extra):
    ''' Map a set of values to client nodes and process them in parallel with `process`. If MPI
    is not enabled, it will use multi-process or serial code depending on the parameters.
    
    .. todo:: figure out what is wrong here!
    
    :Parameters:
    
    process : function
              Function for processing each input value
    vals : list
           List of input values
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    rank : int
           Rank of current node
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    index : int
            Index of the input value in the original list
    res : object
          Result from `process`
    '''
    
    if rank is None: rank = get_rank(comm)
    size = get_size(comm)
    lenbuf = numpy.zeros((size, 1), dtype=numpy.int32)
    _logger.debug("processing - started: %d - %d"%(len(vals), size))
    mpi_type = mpi_dtype(lenbuf.dtype) if MPI is not None else None
    if is_client(comm):
        if rank > 0:
            vals = parallel_utility.partition_list(vals, size-1)
            offset = 1
            for v in vals[:rank-1]: offset += len(v) # 
            vals = vals[rank-1]
        _logger.debug("client-processing - started: %d"%len(vals))
        try:
            for index, res in process_tasks.process_mp(process, vals, **extra):
                #_logger.debug("client-processing: %d of %d-%d -- %d"%(index, rank, size-1,offset))
                if len(vals) == 0: raise ValueError, "An unknown error as occurred"
                if rank > 0:
                    index += offset
                    lenbuf[0, 0] = index
                    #_logger.debug("client-send-1: %d"%rank)
                    comm.Send([lenbuf[0, :], mpi_type], dest=0, tag=4)
                    #_logger.debug("client-send-2: %d"%rank)
                    comm.send(res, dest=0, tag=5)
                    #_logger.debug("client-recv-3: %d"%rank)
                    status = comm.recv(source=0, tag=6)
                    #_logger.debug("client-done-4: %d"%rank)
                    if status < 0: raise StandardError, "Some MPI process crashed"
                else: index += 1
                assert(index>0)
                yield index-1, res
        except:
            _logger.exception("client-processing - error")
            if rank > 0: 
                lenbuf[0, 0] = -1.0
                comm.Send([lenbuf[0, :], mpi_type], dest=0, tag=4)
            raise
        else:
            if rank > 0: 
                lenbuf[0, 0] = 0.0
                comm.Send([lenbuf[0, :], mpi_type], dest=0, tag=4)
            _logger.debug("client-processing - finished")
    else:
        _logger.debug("Root progress monitor - started: %d"%(len(vals)))
        reqs=[]
        node_req=[]
        for i in xrange(1, size):
            reqs.append(comm.Irecv([lenbuf[i, :], mpi_type], source=i, tag=4))
            node_req.append(i)
        status=0
        while len(reqs) > 0:
            idx = MPI.Request.Waitany(reqs)
            node = node_req[idx]
            #_logger.debug("Root root - irecv: %d, %d, %d - status: %d"%(idx, node, lenbuf[node, 0], status))
            if lenbuf[node, 0] > 0:
                #_logger.debug("root-recv-1: %d"%node)
                res = comm.recv(source=node, tag=5)
                #_logger.debug("root-recv-2: %d"%node)
                
                yield int(lenbuf[node, 0])-1, res
                if len(vals) == 0:
                    status=-1
                #_logger.debug("root-send-2: %d"%node)
                comm.send(status, dest=node, tag=6)
                if status == 0:
                    reqs[idx] = comm.Irecv(lenbuf[node, :], source=node, tag=4)
                else:
                    del reqs[idx]
                    del node_req[idx]
                #_logger.debug("root-done-3: %d"%node)
            else:
                _logger.debug("Requesting client shutdown: %d"%node)
                if lenbuf[node, 0] < 0 and status == 0: status=-1
                del reqs[idx]
                del node_req[idx]
        if status < 0: raise ValueError, "Exceptoin raised"
        _logger.debug("Root progress monitor - finished")

def is_root(comm=None, **extra):
    ''' Test if node is root
    
    :Parameters:
    
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    test : bool
           True if rank == 0 or serial code
    '''
    
    return comm is None or comm.Get_rank() == 0 #312/

def is_client(comm=None, **extra):
    ''' Test if node is client
    
    :Parameters:
    
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    test : bool
           True if rank > 0 or serial code
    '''
    
    return comm is None or comm.Get_rank() > 0

def is_client_strict(comm=None, **extra):
    ''' Test if node is client
    
    :Parameters:
    
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    test : bool
           True if rank > 0
    '''
    
    return comm is not None and comm.Get_rank() > 0

def barrier(comm=None, **extra):
    ''' Ensure all node reach this place at the same time
    
    :Parameters:
    
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    extra : dict
            Unused keyword arguments
    '''
    
    if comm is not None: comm.barrier()

def get_size(comm=None, **extra):
    ''' Get number of nodes
    
    :Parameters:
    
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    
    :Returns:
    
    size : int
           Number of nodes
    '''
    
    return 0 if comm is None else comm.Get_size()

def safe_tempfile(filename=None, shmem=True, **extra):
    ''' Return a temp file safe for mutli-processing
    
    :Parameters:
    
    filename : str
               File name template
    shmem : bool
            Use a shared memory mount if exists
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    filename : str
               Temp filename
    '''
    
    id = get_offset(**extra)
    if filename is None:
        filename = "temp_filename.dat"
    base, ext = os.path.splitext(filename)
    
    filename = base+("_%d"%id)+ext
    if shmem:
        shm = os.path.join('/', 'dev', 'shm')
        if os.path.exists(shm): filename = os.path.join(shm, os.path.basename(filename))
    return filename
    

def get_offset(comm=None, process_number=0, **extra):
    ''' Get the current rank and process offset
    
    :Parameters:
    
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    process_number : int
                     Current process number
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    size : int
           Number of nodes
    '''
    
    return get_size(comm)*get_rank(comm) + process_number

def get_rank(comm=None, use_MPI=False, **extra):
    ''' Get rank of current node
    
    :Parameters:
    
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    use_MPI : bool
              If True, create world communicator and check rank
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    rank : int
           Rank of current node
    '''
    
    if comm is None and use_MPI and MPI is not None: comm = MPI.COMM_WORLD
    if comm is not None: return comm.Get_rank()
    return 0

   