''' MPI utility functions

.. Created on Jul 17, 2011
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy, logging
import parallel_utility
import process_tasks
try:
    MPI=None
    from mpi4py import MPI
except:
    logging.warn("MPI not loaded, please install mpi4py")

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

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
    
    if comm is None: return
    rank = comm.Get_rank()
    if rank == 0:
        comm.Recv([data, MPI.DOUBLE], source=root, tag=4)
    elif rank == root:
        comm.Send([data, MPI.DOUBLE], dest=0, tag=4)

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

def block_reduce(data, batch_size=100000, root=0, comm=None, **extra):
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
    data = data.ravel()
    batch_count = (data.shape[0]-1) / batch_size + 1
    block_end = 0
    rank = comm.Get_rank()
    for batch in xrange(batch_count):
        block_beg = block_end
        block_end = min(block_beg+batch_size, data.shape[0])
        if rank == root:
            comm.Reduce(MPI.IN_PLACE, [data[block_beg:block_end], MPI.FLOAT], op=MPI.SUM, root=root)
        else:
            comm.Reduce([data[block_beg:block_end], MPI.FLOAT], None, op=MPI.SUM, root=root)
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
        MPI.COMM_SELF.Set_errhandler(MPI.ERRORS_ARE_FATAL) 
        MPI.COMM_WORLD.Set_errhandler(MPI.ERRORS_ARE_FATAL)
        params['comm'] = comm
        params['rank'] = rank
    elif use_MPI:
        raise ValueError, "MPI failed to initlize - please install mpi4py"

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
    if is_client(comm):
        if rank > 0:
            vals = parallel_utility.partition_array(vals, size-1)
            offset = 1
            for v in vals[:rank-1]: offset += len(v) # 
            vals = vals[rank-1]
        try:
            for index, res in process_tasks.process_mp(process, vals, **extra):
                _logger.debug("client-processing: %d of %d-%d -- %d"%(index, rank, size-1,offset))
                if rank > 0:
                    index += offset
                    lenbuf[0, 0] = index
                    _logger.debug("client-send-1: %d"%rank)
                    comm.Send([lenbuf[0, :], MPI.INT], dest=0, tag=4)
                    _logger.debug("client-send-2: %d"%rank)
                    comm.send(res, dest=0, tag=5)
                    _logger.debug("client-recv-3: %d"%rank)
                    status = comm.recv(source=0, tag=6)
                    _logger.debug("client-done-4: %d"%rank)
                    if status < 0: raise StandardError, "Some MPI process crashed"
                else: index += 1
                assert(index>0)
                yield index-1, res
        except:
            _logger.debug("client-processing - error")
            if rank > 0: 
                lenbuf[0, 0] = -1.0
                comm.Send([lenbuf[0, :], MPI.INT], dest=0, tag=4)
            raise
        else:
            if rank > 0: 
                lenbuf[0, 0] = 0.0
                comm.Send([lenbuf[0, :], MPI.INT], dest=0, tag=4)
            _logger.debug("client-processing - finished")
    else:
        _logger.debug("Root progress monitor - started")
        reqs=[]
        node_req=[]
        for i in xrange(1, size):
            reqs.append(comm.Irecv([lenbuf[i, :], MPI.INT], source=i, tag=4))
            node_req.append(i)
        status=0
        while len(reqs) > 0:
            idx = MPI.Request.Waitany(reqs)
            node = node_req[idx]
            _logger.debug("Root root - irecv: %d, %d, %d - status: %d"%(idx, node, lenbuf[node, 0], status))
            if lenbuf[node, 0] > 0:
                _logger.debug("root-recv-1: %d"%node)
                res = comm.recv(source=node, tag=5)
                _logger.debug("root-recv-2: %d"%node)
                
                yield int(lenbuf[node, 0])-1, res
                if len(vals) == 0:
                    status=-1
                _logger.debug("root-send-2: %d"%node)
                comm.send(status, dest=node, tag=6)
                if status == 0:
                    reqs[idx] = comm.Irecv(lenbuf[node, :], source=node, tag=4)
                else:
                    del reqs[idx]
                    del node_req[idx]
                _logger.debug("root-done-3: %d"%node)
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

def get_rank(use_MPI=False, comm=None, **extra):
    ''' Get rank of current node
    
    :Parameters:
    
    use_MPI : bool
              If True, create world communicator and check rank
    comm : mpi4py.MPI.Intracomm
           MPI communications object
    
    :Returns:
    
    rank : int
           Rank of current node
    '''
    
    if comm is None and use_MPI and MPI is not None: comm = MPI.COMM_WORLD
    if comm is not None: return comm.Get_rank()
    return 0

   