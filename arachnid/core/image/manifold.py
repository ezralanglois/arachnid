''' Manifold learning algorithms

.. Created on Oct 24, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
import logging, numpy, scipy
import scipy.linalg, scipy.io
import scipy.sparse.linalg
from ..metadata import format_utility
from ..parallel import process_queue #, openmp

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from util import _manifold
    _manifold;
except:
    if _logger.isEnabledFor(logging.DEBUG):
        tracing.log_import_error('Failed to load _manifold.so module - certain functions will not be available', _logger)
    try:
        import _manifold
        _manifold;
    except:
        tracing.log_import_error('Failed to load _manifold.so module - certain functions will not be available', _logger)
        _manifold = None

        
def diffusion_maps(samp, dimension, k, mutual=True, batch=10000):
    ''' Embed the sample data into a low dimensional manifold using the
    Diffusion Maps algorithm.
    
    :Parameters:
    
    samp : array
           2D data array
    dimension : int
                Number of dimensions for embedding
    k : int
        Number of nearest neighbors
    mutual : bool
             Keep only mutal neighbors
    batch : int
            Size of temporary distance matrix to hold in memory
    
    :Returns:
    
    evecs : array
            2D array of eigen vectors
    evals : array
            1D array of eigen values
    index : array
            Index array if a subset of from the largest connected 
            component is used, otherwise None
    '''
    
    dist2 = knn(samp, k, batch)
    assert(dist2.shape[0]==samp.shape[0])
    dist2 = knn_reduce(dist2, k, mutual)
    return diffusion_maps_dist(dist2, dimension)

def knn_reduce_eps(dist2, eps, epsdata=None):
    '''Reduce k-nearest neighbor sparse matrix
    in the COO format to one with less neighbors.
    
    :Parameters:
    
    dist2 : sparse array
            Sparse distance matrix in COO format
    eps : float
          Maximum allowed distance between neighbors
    epsdata : array, optional
              Data vector used for comparison (not copied)
                
    :Returns:
    
    dist2 : sparse array
            Sparse distance matrix in COO format
    '''
    
    data = dist2.data.copy()
    row  = dist2.row.copy()
    col  = dist2.col.copy()
    if epsdata is not None:
        n = _manifold.knn_reduce_eps_cmp(dist2.data, dist2.col, dist2.row, data, col, row, epsdata, eps)
    else:
        n = _manifold.knn_reduce_eps(dist2.data, dist2.col, dist2.row, data, col, row, eps)
    _logger.error("n=%d from %d"%(n, dist2.data.shape[0]))
    data = data[:n]
    col = col[:n]
    row = row[:n]
    return scipy.sparse.coo_matrix( (data,(row, col)), shape=dist2.shape )

def knn_reduce(dist2, k, mutual=False):
    '''Reduce k-nearest neighbor sparse matrix
    in the COO format to one with less neighbors.
    
    :Parameters:
    
    dist2 : sparse array
            Sparse distance matrix in COO format
    k : int
        Number of nearest neighbors
    mutual : bool
             Keep only mutal neighbors
    
    :Returns:
    
    dist2 : sparse array
            Sparse distance matrix in COO format
    '''
    
    k+=1 # include self as a neighbor
    n = dist2.shape[0]*k if mutual else dist2.shape[0]*k*2
    data = numpy.empty(n, dtype=dist2.data.dtype)
    row  = numpy.empty(n, dtype=dist2.row.dtype)
    col  = numpy.empty(n, dtype=dist2.col.dtype)
    d = dist2.data.shape[0]/dist2.shape[0] - k
    if d < 0: raise ValueError, "Cannot reduce from %d neighbors to %d"%(dist2.data.shape[0]/dist2.shape[0], k)
    if d > 0:
        _manifold.knn_reduce(dist2.data, dist2.col, dist2.row, data, col, row, d, k)
    else:
        data[:dist2.data.shape[0]] = dist2.data
        row[:dist2.row.shape[0]] = dist2.row
        col[:dist2.col.shape[0]] = dist2.col
        _logger.error("col1: %s"%str(dist2.col[:10]))
        _logger.error("col2: %s"%str(col[:10]))
        
    if not mutual:
        m = dist2.shape[0]*k
        data[m:] = data[:m]
        row[m:] = row[:m]
        col[m:] = col[:m]
    else:
        n=_manifold.knn_mutual(data, col, row, k)
        _logger.error("No reduction: k=%d, n=%d"%(k, n))
        data = data[:n]
        col = col[:n]
        row = row[:n]
    _logger.error("len -> %d == %d"%(data.shape[0], dist2.data.shape[0]))
    return scipy.sparse.coo_matrix( (data,(row, col)), shape=dist2.shape )

def knn_simple(samp, k, dtype=numpy.float):
    ''' Calculate k-nearest neighbors and store in a sparse matrix
    in the COO format.
    
    :Parameters:
    
    samp : array
           2D data array
    k : int
        Number of nearest neighbors
    batch : int
            Size of temporary distance matrix to hold in memory
    
    :Returns:
    
    dist2 : sparse array
            Sparse distance matrix in COO format
    '''
    import scipy.spatial
    
    k+=1 # include self as a neighbor
    n = samp.shape[0]*k
    data = numpy.empty(n, dtype=dtype)
    col = numpy.empty(n, dtype=numpy.longlong)
    dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(samp, 'sqeuclidean'))
    
    for r in xrange(samp.shape[0]):
        idx = numpy.argsort(dist[r, :])
        data[r*k:(r+1)*k] = dist[r, idx[:k]]
        col[r*k:(r+1)*k] = idx[:k]
    
    row = numpy.empty(n, dtype=numpy.longlong)
    for r in xrange(samp.shape[0]):
        row[r*k:(r+1)*k]=r
    return scipy.sparse.coo_matrix((data,(row, col)), shape=(samp.shape[0], samp.shape[0]))

def local_neighbor_average(samp, neigh):
    ''' Create a set of locally averaged samples
    
    :Parameters:
    
    samp : array
           Sample
    neigh : sparse matrix
            Neighborhood
    
    :Returns:
    
    avg_samp : array
               Locally averaged sample
    '''
    
    avgsamp = numpy.empty_like(samp)
    mval = numpy.mod(samp.shape[0], neigh.col.shape[0]) 
    if mval != 0 and mval != samp.shape[0]: raise ValueError, "Sample array does not match neighborhood: %d mod %d = %d"%(samp.shape[0], neigh.col.shape[0], mval )
    neighbors = neigh.col.shape[0]/samp.shape[0]
    b = 0
    for i in xrange(samp.shape[0]):
        e = b+neighbors
        assert( neigh.row[b] == neigh.col[b] )
        avgsamp[i] = numpy.mean(samp[neigh.col[b:e]])
        b=e
    return avgsamp

"""
def load_from_cache(cache_file, n, suffix=""):
    '''
    '''
    
    if cache_file is None: return None
    
    cache_file = format_utility.new_filename(cache_file, ext=".mat")
    cache_dat = format_utility.new_filename(cache_file, suffix="_coo"+suffix, ext=".bin")
    cache_row = format_utility.new_filename(cache_file, suffix="_row"+suffix, ext=".bin")
    cache_col = format_utility.new_filename(cache_file, suffix="_col"+suffix, ext=".bin")
    if format_utility.os.path.exists(cache_file):
        mat = scipy.io.loadmat(cache_file)
        
        if mat['coo'][0] == samp.shape[0] and mat['total'] == n:
            dtype_coo = mat['dtype_coo'][0]
            dtype_col = mat['dtype_col'][0]
            if dtype_coo[0] == '[': dtype_coo = dtype_coo[2:len(dtype_coo)-2]
            if dtype_col[0] == '[': dtype_col = dtype_col[2:len(dtype_col)-2]
            coo = numpy.fromfile(cache_dat, dtype=numpy.dtype(dtype_coo))
            row = numpy.fromfile(cache_row, dtype=numpy.dtype(dtype_col))
            col = numpy.fromfile(cache_col, dtype=numpy.dtype(dtype_col))
            return scipy.sparse.coo_matrix( (coo.squeeze(), (row.squeeze(), col.squeeze())), shape=tuple(mat['coo'])) #(mat['coo'][0], mat['coo'][1]) )
    return None
"""

def eps_range(neigh, neighbors=None):
    ''' Estimate the epsilon nearest neighbor range for a given range of neighbors
    
    :Parameters:
    
    neigh : sparse matrix
            Sparse matrix with nearest neighbors
    neighbors : int, optional
                Number of nearest neighbors
                
    :Returns:
    
    eps_max : flaot
              Maximum EPS value
    eps_min : float
              Minimum EPS value
    '''
    
    if neighbors is None:
        neighbors = neigh.data.shape[0]/neigh.shape[0]
    else: neighbors += 1
    
    data = neigh.data.reshape((neigh.shape[0], neighbors))
    maxvals = data.max(axis=1)
    eps_max = maxvals.min()
    maxvals = data.min(axis=1)
    eps_min = maxvals.max()
    
    return eps_max, eps_min

def knn_geodesic_cache(samp, k, batch=10000, shared=False, cache_file=None):
    ''' Calculate k-nearest neighbors and store in a sparse matrix
    in the COO format or load from a pre-calculated file.
    
    :Parameters:
    
    samp : array
           2D data array
    k : int
        Number of nearest neighbors
    batch : int
            Size of temporary distance matrix to hold in memory
    
    :Returns:
    
    dist2 : sparse array
            Sparse distance matrix in COO format
    '''
    
    if cache_file is not None:
        cache_file = format_utility.new_filename(cache_file, ext=".mat")
        cache_dat = format_utility.new_filename(cache_file, suffix="_coo", ext=".bin")
        cache_row = format_utility.new_filename(cache_file, suffix="_row", ext=".bin")
        cache_col = format_utility.new_filename(cache_file, suffix="_col", ext=".bin")
        if format_utility.os.path.exists(cache_file):
            mat = scipy.io.loadmat(cache_file)
            
            if mat['coo'][0] == samp.shape[0] and mat['total'] == ((k+1)*samp.shape[0]):
                dtype_coo = mat['dtype_coo'][0]
                dtype_col = mat['dtype_col'][0]
                if dtype_coo[0] == '[': dtype_coo = dtype_coo[2:len(dtype_coo)-2]
                if dtype_col[0] == '[': dtype_col = dtype_col[2:len(dtype_col)-2]
                if shared:
                    n = mat['total']
                    data, shm_data = process_queue.create_global_dense_matrix((n, ), numpy.dtype(dtype_coo), shared)
                    col, shm_col = process_queue.create_global_dense_matrix((n, ), numpy.dtype(dtype_col), shared)
                    row, shm_row = process_queue.create_global_dense_matrix((n, ), numpy.dtype(dtype_col), shared)
                    data[:] = numpy.fromfile(cache_dat, dtype=numpy.dtype(dtype_coo))
                    row[:] = numpy.fromfile(cache_row, dtype=numpy.dtype(dtype_col))
                    col[:] = numpy.fromfile(cache_col, dtype=numpy.dtype(dtype_col))
                    smat = scipy.sparse.coo_matrix( (data.squeeze(), (row.squeeze(), col.squeeze())), shape=tuple(mat['coo'])) #(mat['coo'][0], mat['coo'][1]) ) 
                    smat.shmem = (shm_data, shm_row, shm_col, tuple(mat['coo']))
                    return smat
                else:
                    coo = numpy.fromfile(cache_dat, dtype=numpy.dtype(dtype_coo))
                    row = numpy.fromfile(cache_row, dtype=numpy.dtype(dtype_col))
                    col = numpy.fromfile(cache_col, dtype=numpy.dtype(dtype_col))
                    return scipy.sparse.coo_matrix( (coo.squeeze(), (row.squeeze(), col.squeeze())), shape=tuple(mat['coo'])) #(mat['coo'][0], mat['coo'][1]) )    
    mat = knn_geodesic(samp, k, batch, shared)
    if cache_file is not None:
        scipy.io.savemat(cache_file, dict(coo=numpy.asarray(mat.shape, dtype=mat.col.dtype), total=mat.data.shape[0], dtype_coo=mat.data.dtype.name, dtype_col=mat.col.dtype.name), oned_as='column', format='5')
        mat.data.tofile(cache_dat)
        mat.row.tofile(cache_row)
        mat.col.tofile(cache_col)
    return mat
    

def knn_geodesic(samp, k, batch=10000, shared=False):
    ''' Calculate k-nearest neighbors and store in a sparse matrix
    in the COO format.
    
    :Parameters:
    
    samp : array
           2D data array
    k : int
        Number of nearest neighbors
    batch : int
            Size of temporary distance matrix to hold in memory
    
    :Returns:
    
    dist2 : sparse array
            Sparse distance matrix in COO format
    '''
    
    if _manifold is None: raise ValueError('Failed to import manifold')
    if samp.ndim != 2: raise ValueError('Expects 2D array')
    if samp.shape[1] != 4: raise ValueError('Expects matrix of quaternions')
    
    dtype = samp.dtype
    k = int(k)
    k+=1
    n = samp.shape[0]*k
    nbatch = int(samp.shape[0]/float(batch))
    batch = int(samp.shape[0]/float(nbatch))
    
    data, shm_data = process_queue.create_global_dense_matrix(n, dtype, shared)
    col, shm_col = process_queue.create_global_dense_matrix(n, numpy.longlong, shared)
    dense = numpy.empty((batch,batch), dtype=data.dtype)
    
    #gemm = scipy.linalg.fblas.dgemm
    for r in xrange(0, samp.shape[0], batch):
        rnext = min(r+batch, samp.shape[0])
        beg, end = r*k, rnext*k
        s2 = samp[r:rnext]
        for c in xrange(0, samp.shape[0], batch):
            s1 = samp[c:min(c+batch, samp.shape[0])]
            tmp = dense.ravel()[:s1.shape[0]*s2.shape[0]].reshape((s2.shape[0],s1.shape[0]))
            #dist2 = gemm(1.0, s1.T, s2.T, trans_a=True, beta=0, c=tmp.T, overwrite_c=1).T
            _manifold.gemm(s1, s2, tmp, 1.0, 0.0)
            dist2=tmp
            dist2[dist2>1.0]=1.0
            numpy.arccos(dist2, dist2)
            _logger.error("push_to_heap-1")
            _manifold.push_to_heap(dist2, data[beg:], col[beg:], int(c), k)
            _logger.error("push_to_heap-2")
        _logger.error("finalize_heap-1")
        _manifold.finalize_heap(data[beg:end], col[beg:end], int(r), k)
        _logger.error("finalize_heap-2")
    
    _logger.error("del-1")
    del dist2, dense
    _logger.error("del-2")
    row, shm_row = process_queue.create_global_dense_matrix(n, numpy.longlong, shared)
    rtmp = row.reshape((samp.shape[0], k))
    ctmp = col.reshape((samp.shape[0], k))
    dtmp = data.reshape((samp.shape[0], k))
    for r in xrange(samp.shape[0]):
        rtmp[r, :]=r
        if ctmp[r, 0]!=r:
            _logger.error("%d == %d --> %f"%(ctmp[r, 0], r, dtmp[r, 0]))
        assert(ctmp[r, 0]==r)
    mat = scipy.sparse.coo_matrix((data,(row, col)), shape=(samp.shape[0], samp.shape[0]))
    mat.shmem = (shm_data, shm_row, shm_col, (samp.shape[0], samp.shape[0]))
    return mat

def knn(samp, k, batch=10000):
    ''' Calculate k-nearest neighbors and store in a sparse matrix
    in the COO format.
    
    :Parameters:
    
    samp : array
           2D data array
    k : int
        Number of nearest neighbors
    batch : int
            Size of temporary distance matrix to hold in memory
    
    :Returns:
    
    dist2 : sparse array
            Sparse distance matrix in COO format
    '''
    
    k+=1 # include self as a neighbor
    n = samp.shape[0]*k
    dtype = samp.dtype
    nbatch = max(1, int(samp.shape[0]/float(batch)))
    batch = int(samp.shape[0]/float(nbatch))
    data = numpy.empty(n, dtype=dtype)
    col = numpy.empty(n, dtype=numpy.longlong)
    dense = numpy.empty((batch,batch), dtype=dtype)
    
    #gemm = scipy.linalg.fblas.dgemm
    a = (samp**2).sum(axis=1)
    assert(a.shape[0]==samp.shape[0])
    for r in xrange(0, samp.shape[0], batch):
        rnext = min(r+batch, samp.shape[0])
        beg, end = r*k, rnext*k
        s2 = samp[r:rnext]
        for c in xrange(0, samp.shape[0], batch):
            s1 = samp[c:min(c+batch, samp.shape[0])]
            tmp = dense.ravel()[:s1.shape[0]*s2.shape[0]].reshape((s2.shape[0],s1.shape[0]))
            #dist2 = numpy.dot(s1, s2.T)
            #dist2 *= -2.0
            _manifold.gemm(s1, s2, tmp, -2.0, 0.0)
            dist2=tmp
            #dist2 = gemm(-2.0, s1.T, s2.T, trans_a=True, beta=0, c=tmp.T, overwrite_c=1).T
            #dist2 = gemm(-2.0, s1, s2, trans_b=True, beta=0, c=tmp, overwrite_c=1).T
            try:
                dist2 += a[c:c+batch]#, numpy.newaxis]
            except:
                _logger.error("dist2.shape=%s -- a.shape=%s -- %d,%d,%d"%(str(dist2.shape), str(a.shape), c, c+batch, batch))
                raise
            dist2 += a[r:rnext, numpy.newaxis]
            _manifold.push_to_heap(dist2, data[beg:], col[beg:], int(c), k)
        _manifold.finalize_heap(data[beg:end], col[beg:end], int(r), k)
    del dist2, dense
    row = numpy.empty(n, dtype=numpy.longlong)
    tmp = row.reshape((samp.shape[0], k))
    for r in xrange(samp.shape[0]):
        tmp[r, :]=r
    return scipy.sparse.coo_matrix((data,(row, col)), shape=(samp.shape[0], samp.shape[0]))

def euclidean_distance2(X, Y):
    ''' Calculate the squared euclidean distance between two 
    matrices using optimized BLAS
    
    :Parameters:
    
    X : array
        Matrix 1 (mxa)
    Y : array
        Matrix 2 (nxa)
    
    :Returns:
    
    dist2 : array
            Array holding the distance between both matrices (mxn)
    '''
    
    gemm = scipy.linalg.fblas.dgemm
    x2 = (X**2).sum(axis=1)
    y2 = (Y**2).sum(axis=1)
    dist2 = gemm(-2.0, X, Y, trans_b=True, beta=0).T
    dist2 += x2[:, numpy.newaxis]
    dist2 += y2
    return dist2

def diffusion_maps_dist(dist2, dimension):
    ''' Embed a sparse distance matrix with the diffusion maps 
    manifold learning algorithm
    
    :Parameters:
    
    dist2 : sparse matrix (coo, csr, csc)
            Sparse symmetric distance matrix to embed
    dimension : int
                Number of dimensions for embedding
    
    :Returns:
    
    evecs : array
            2D array of eigen vectors
    evals : array
            1D array of eigen values
    index : array
            Index array if a subset of from the largest connected 
            component is used, otherwise None
    '''
    
    if not scipy.sparse.isspmatrix_csr(dist2): dist2 = dist2.tocsr()
    dist2, index = largest_connected(dist2)
    _manifold.self_tuning_gaussian_kernel_csr(dist2.data, dist2.data, dist2.indices, dist2.indptr)
    _manifold.normalize_csr(dist2.data, dist2.data, dist2.indices, dist2.indptr) # problem here
    assert(numpy.alltrue(numpy.isfinite(dist2.data)))
    D = scipy.power(dist2.sum(axis=0)+1e-12, -0.5)
    assert(numpy.alltrue(numpy.isfinite(D)))
    #D[numpy.logical_not(numpy.isfinite(D))] = 1.0
    norm = scipy.sparse.dia_matrix((D, (0,)), shape=dist2.shape)
    L = norm * dist2 * norm
    del norm
    #openmp.set_thread_count(1)
    #assert(numpy.alltrue(numpy.isfinite(L.data)))
    if hasattr(scipy.sparse.linalg, 'eigen_symmetric'):
        evals, evecs = scipy.sparse.linalg.eigen_symmetric(L, k=dimension+1)
    else:
        try:
            evals, evecs = scipy.sparse.linalg.eigsh(L, k=dimension+1)
        except:
            _logger.error("%s --- %d"%(str(L.shape), dimension))
            raise
    del L
    evecs, D = numpy.asarray(evecs), numpy.asarray(D).squeeze()
    index2 = numpy.argsort(evals)[-2:-2-dimension:-1]
    evecs = D[:,numpy.newaxis] * evecs #[:, index2]
    return evecs[:, index2], evals[index2].squeeze(), index
    
def largest_connected(dist2):
    ''' Find the largest connected component in a graph
    
    :Parameters:
    
    dist2 : sparse matrix (coo, csr, csc)
            Sparse symmetric distance matrix to test
    
    :Returns:
    
    dist2 : sparse matrix (coo, csr, csc)
            Largest connected component from the sparse matrix
    index : array
            Rows kept from the original matrix
    '''
    
    tmp = dist2.tocsr() if not scipy.sparse.isspmatrix_csr(dist2) else dist2
    #tmp = (tmp + tmp.T)/2
    n, comp = scipy.sparse.csgraph.cs_graph_components(tmp, connection='strong')
    b = len(numpy.unique(comp))
    _logger.info("Check connected components: %d -- %d -- %d (%d)"%(n, b, dist2.shape[0], dist2.data.shape[0]))
    n = b
    index = None
    if n > 1:
        bins = numpy.unique(comp)
        count = numpy.zeros(len(bins))
        for i in xrange(len(bins)):
            if bins[i] > -1:
                count[i] = numpy.sum(comp == bins[i])
        _logger.info("Number of components: %d"%(n))
        if 1 == 1:
            idx = numpy.argsort(count)[::-1]
            for i in idx[:10]:
                _logger.info("%d: %d"%(bins[i], count[i]))
        index = numpy.argwhere(comp == bins[numpy.argmax(count)]).ravel()
        index = index.astype(dist2.indices.dtype)
        n=_manifold.select_subset_csr(dist2.data, dist2.indices, dist2.indptr, index)
        dist2=scipy.sparse.csr_matrix((dist2.data[:n],dist2.indices[:n], dist2.indptr[:index.shape[0]+1]), shape=(index.shape[0], index.shape[0]))
    assert(scipy.sparse.csgraph.cs_graph_components(dist2.tocsr())[0]==1)
    return dist2, index


