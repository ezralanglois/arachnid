''' Manifold learning algorithms

.. Created on Oct 24, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
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
    from ..app import tracing
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
    
    n = dist2.data.shape[0]*2
    data = numpy.empty(n, dtype=dist2.data.dtype)
    row  = numpy.empty(n, dtype=dist2.row.dtype)
    col  = numpy.empty(n, dtype=dist2.col.dtype)
    if epsdata is not None:
        n = _manifold.knn_reduce_eps_cmp(dist2.data, dist2.col, dist2.row, data, col, row, epsdata, eps)
    else:
        n = _manifold.knn_reduce_eps(dist2.data, dist2.col, dist2.row, data, col, row, eps)
    _logger.error("n=%d from %d"%(n, dist2.data.shape[0]))
    data[n:2*n] = data[:n]
    row[n:2*n] = row[:n]
    col[n:2*n] = col[:n]
    data = data[:2*n]
    col = col[:2*n]
    row = row[:2*n]
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
    l = dist2.shape[0]*k
    data = numpy.empty(n, dtype=dist2.data.dtype)
    row  = numpy.empty(n, dtype=dist2.row.dtype)
    col  = numpy.empty(n, dtype=dist2.col.dtype)
    d = dist2.data.shape[0]/dist2.shape[0] - k
    if d < 0: raise ValueError, "Cannot reduce from %d neighbors to %d"%(dist2.data.shape[0]/dist2.shape[0], k)
    if d > 0:
        _manifold.knn_reduce(dist2.data, dist2.col, dist2.row, data[:l], col[:l], row[:l], d, k)
    else:
        data[:dist2.data.shape[0]] = dist2.data
        row[:dist2.row.shape[0]] = dist2.row
        col[:dist2.col.shape[0]] = dist2.col

    if not mutual:
        m = dist2.shape[0]*k
        data[m:] = data[:m]
        row[m:] = row[:m]
        col[m:] = col[:m]
    else:
        n=_manifold.knn_mutual(data, col, row, k)
        data = data[:n]
        col = col[:n]
        row = row[:n]
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

def local_neighbor_average(samp, neigh, subset=None):
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
    
    avgsamp = numpy.empty_like(samp) if subset is None else numpy.zeros((subset.shape[0], samp.shape[1]), dtype=samp.dtype)
    mval = numpy.mod(samp.shape[0], neigh.col.shape[0]) 
    if mval != 0 and mval != samp.shape[0]: raise ValueError, "Sample array does not match neighborhood: %d mod %d = %d"%(samp.shape[0], neigh.col.shape[0], mval )
    neighbors = neigh.col.shape[0]/samp.shape[0]
    if subset is not None:
        b = 0
        for i in xrange(subset.shape[0]):
            assert( neigh.row[b] == neigh.col[b] )
            b = subset[i]*neighbors
            e = b+neighbors
            avgsamp[i] = numpy.mean(samp[neigh.col[b:e]])
    else:
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
    
    k = int(k)
    k+=1
    n = samp.shape[0]*k
    dtype = samp.dtype
    nbatch = max(1, int(samp.shape[0]/float(batch)))
    batch = int(samp.shape[0]/float(nbatch))
    data = numpy.empty(n, dtype=dtype)
    col = numpy.empty(n, dtype=numpy.longlong)
    dense = numpy.empty((batch,batch), dtype=dtype)
    
    #gemm = scipy.linalg.fblas.dgemm
    for r in xrange(0, samp.shape[0], batch):
        rnext = min(r+batch, samp.shape[0])
        beg, end = r*k, rnext*k
        s1 = samp[r:rnext]
        for c in xrange(0, samp.shape[0], batch):
            s2 = samp[c:min(c+batch, samp.shape[0])]
            tmp = dense.ravel()[:s1.shape[0]*s2.shape[0]].reshape((s1.shape[0],s2.shape[0]))
            #dist2 = gemm(1.0, s1.T, s2.T, trans_a=True, beta=0, c=tmp.T, overwrite_c=1).T
            _manifold.gemm(s1, s2, tmp, 1.0, 0.0)
            dist2=tmp
            dist2[dist2>1.0]=1.0
            numpy.arccos(dist2, dist2)
            # assert dist2 < pi
            _manifold.push_to_heap(dist2, data[beg:], col[beg:], int(c), k)
        _manifold.finalize_heap(data[beg:end], col[beg:end], int(r), k)
    
    del dist2, dense
    row = numpy.empty(n, dtype=numpy.longlong)
    rtmp = row.reshape((samp.shape[0], k))
    for r in xrange(samp.shape[0]):
        rtmp[r, :]=r
    return scipy.sparse.coo_matrix((data,(row, col)), shape=(samp.shape[0], samp.shape[0]))

def fastdot_t2(s1, s2_t, out=None, alpha=1.0, beta=0.0):
    '''
    '''
    
    if s1.shape[1] != s2_t.shape[1]: raise ValueError, "Number of columns in each matrix does not match: %d != %d"%(s1.shape[1], s2_t.shape[1])
    if out is None: out = numpy.zeros((s1.shape[0], s2_t.shape[0]), dtype=s1.dtype)
    _manifold.gemm(s1, s2_t, out, float(alpha), float(beta))
    return out

def fastdot_t1(s1_t, s2, out=None, alpha=1.0, beta=0.0):
    '''
    '''
    
    if s1_t.shape[0] != s2.shape[0]: raise ValueError, "Number of columns in each matrix does not match: %d != %d"%(s1_t.shape[1], s2.shape[1])
    if out is None: out = numpy.zeros((s1_t.shape[1], s2.shape[1]), dtype=s1_t.dtype)
    _manifold.gemm_t1(s1_t, s2, out, float(alpha), float(beta))
    return out

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
    
    k = int(k)
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
        s1 = samp[r:rnext]
        for c in xrange(0, samp.shape[0], batch):
            s2 = samp[c:min(c+batch, samp.shape[0])]
            tmp = dense.ravel()[:s1.shape[0]*s2.shape[0]].reshape((s1.shape[0],s2.shape[0]))
            
            # Using fblas requires you switch the order of the row and col
            #s1 = samp[c:min(c+batch, samp.shape[0])]
            #tmp = dense.ravel()[:s1.shape[0]*s2.shape[0]].reshape((s2.shape[0],s1.shape[0]))
            #dist2 = scipy.linalg.fblas.dgemm(-2.0, s1.T, s2.T, trans_a=True, beta=0, c=tmp.T, overwrite_c=1).T
            
            
            _manifold.gemm(s1, s2, tmp, -2.0, 0.0)
            dist2=tmp
            
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

def self_tuning_gaussian_kernel_dense(dist2, normalize=False, normalize2=False):
    '''
    '''
    
    w1 = numpy.sqrt(dist2.max(axis=0))
    w2 = numpy.sqrt(dist2.max(axis=1))
    for i in xrange(len(dist2)):
        dist2[i] /= -(w1 * w2[i]+1e-12)
    numpy.exp(dist2, dist2)
    if normalize:
        w1 = numpy.sqrt(dist2.sum(axis=0))
        w2 = numpy.sqrt(dist2.sum(axis=1))
        for i in xrange(len(dist2)):
            dist2[i] /= (w1 * w2[i]+1e-12)
    if normalize2:
        Dr = 1./(numpy.sqrt(dist2.sum(axis=0))+1e-12)
        Dc = 1./(numpy.sqrt(dist2.sum(axis=1))+1e-12)
        dist2[:] = Dr * dist2 * Dc[:, numpy.newaxis]
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
    dist2 = (dist2 + dist2.T)/2
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

def reduce_subset(dist2, index):
    ''' Reduce a sparse matrix (CSR format) using the set of selected
    '''
    
    tmp = dist2.tocsr() if not scipy.sparse.isspmatrix_csr(dist2) else dist2
    index = index.astype(tmp.indices.dtype)
    n=_manifold.select_subset_csr(tmp.data, tmp.indices, tmp.indptr, index)
    dist2=scipy.sparse.csr_matrix((tmp.data[:n],tmp.indices[:n], tmp.indptr[:index.shape[0]+1]), shape=(index.shape[0], index.shape[0]))
    return dist2
    
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
    if hasattr(scipy.sparse.csgraph, 'connected_components') and 1 == 0:
        n, comp = scipy.sparse.csgraph.connected_components(tmp)
    else:
        n, comp = scipy.sparse.csgraph.cs_graph_components(tmp)
        
    b = len(numpy.unique(comp))
    _logger.info("Check connected components: %d -- %d -- %d (%d)"%(n, b, tmp.shape[0], tmp.data.shape[0]))
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
        index = index.astype(tmp.indices.dtype)
        n=_manifold.select_subset_csr(tmp.data, tmp.indices, tmp.indptr, index)
        dist2=scipy.sparse.csr_matrix((tmp.data[:n],tmp.indices[:n], tmp.indptr[:index.shape[0]+1]), shape=(index.shape[0], index.shape[0]))
    #assert(scipy.sparse.csgraph.cs_graph_components(dist2.tocsr())[0]==1)
    return dist2, index

