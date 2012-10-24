''' Manifold learning algorithms

.. Created on Oct 24, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
import logging, numpy, scipy
import scipy.linalg
import scipy.sparse.linalg
#from ..parallel import process_queue

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
    
    n = dist2.shape[0]*k if mutual else dist2.shape[0]*k*2
    data = numpy.empty(n, dtype=dist2.data.dtype)
    row  = numpy.empty(n, dtype=dist2.row.dtype)
    col  = numpy.empty(n, dtype=dist2.col.dtype)
    d = dist2.data.shape[0]/dist2.shape[0] - k
    _manifold.knn_reduce(dist2.data, dist2.row, dist2.col, data, row, col, d, k)
    if not mutual:
        m=dist2.shape[0]*k
        data[m:] = data[:m]
        row[m:] = row[:m]
        col[m:] = col[:m]
    else:
        n=_manifold.knn_mutual(data, row, col, k)
        data = data[:n]
        col = col[:n]
        row = row[:n]
    return scipy.sparse.coo_matrix( (data,(row, col)), shape=data.shape )

def knn(samp, k, batch=10000, dtype=numpy.float):
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
    
    k+=1
    n = samp.shape[0]*k
    nbatch = int(samp.shape[0]/float(batch))
    batch = int(samp.shape[0]/float(nbatch))
    data = numpy.empty(n, dtype=dtype)
    col = numpy.empty(n, dtype=numpy.longlong)
    dense = numpy.empty((batch,batch), dtype=dtype)
    
    a = (samp**2).sum(axis=1)
    for r in xrange(0, samp.shape[0], batch):
        for c in xrange(0, samp.shape[0], batch):
            dist2 = gemm(-2.0, X, Y, trans_b=True, beta=0, c=dense, overwrite_c=1).T
            dist2 += a[r:r+batch, numpy.newaxis]
            dist2 += a[c:c+batch]
            _manifold.push_to_heap(dist2, data[r*batch*k:], col[r*batch*k:], c, k)
        _manifold.finalize_heap(data[r*batch*k:], col[r*batch*k:], k)
            
    del dist2, dense
    row = numpy.empty(n, dtype=numpy.longlong)
    for r in xrange(samp.shape[0]):
        row[r*k:(r+1)*k]=r
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
    _manifold.self_tuning_gaussian_kernel_csr(dist2.data, dist2.data, dist2.indices, data2.indptr)
    _manifold.normalize_csr(dist2.data, dist2.data, dist2.indices, data2.indptr)
    D = scipy.power(dist2.sum(axis=0), -0.5)
    D[numpy.logical_not(numpy.isfinite(D))] = 1.0
    norm = scipy.sparse.dia_matrix((D, (0,)), shape=dist2.shape)
    L = norm * dist2 * norm
    del norm
    if hasattr(scipy.sparse.linalg, 'eigen_symmetric'):
        evals, evecs = scipy.sparse.linalg.eigen_symmetric(L, k=dimension+1)
    else:
        evals, evecs = scipy.sparse.linalg.eigsh(L, k=dimension+1)
    del L
    evecs, D = numpy.asarray(evecs), numpy.asarray(D).squeeze()
    index = numpy.argsort(evals)[-2:-2-dimension:-1]
    evecs = D[:,numpy.newaxis] * evecs[:, index]
    return evecs, evals[index].squeeze(), index
    
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
    tmp = (tmp + tmp.T)/2
    n, comp = scipy.sparse.csgraph.cs_graph_components(tmp)
    b = len(numpy.unique(comp))
    _logger.debug("Check connected components: %d -- %d -- %d (%d)"%(n, b, dist2.shape[0], dist2.data.shape[0]))
    n = b
    index = None
    if n > 1:
        bins = numpy.unique(comp)
        count = numpy.zeros(len(bins))
        for i in xrange(len(bins)):
            if bins[i] > -1:
                count[i] = numpy.sum(comp == bins[i])
        _logger.info("Number of components: %d"%(n))
        if 1 == 0:
            idx = numpy.argsort(count)[::-1]
            for i in idx[:10]:
                _logger.info("%d: %d"%(bins[i], count[i]))
        index = numpy.argwhere(comp == bins[numpy.argmax(count)])
        n=_manifold.select_subset_csr(dist2.data, dist2.indices, data2.indptr, index)
        dist2.data=dist2.data[:n]
        dist2.indices=dist2.indices[:n]
        data2.indptr=data2.indptr[:index.shape[0]+1]
    return dist2, index


