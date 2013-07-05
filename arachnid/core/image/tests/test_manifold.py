'''
.. Created on Oct 25, 2012
.. codeauthor:: robertlanglois
'''

import numpy.testing, scipy
from .. import manifold

def test_knn():
    '''
    '''
    
    samp = numpy.random.rand(20,5)
    dist1 = manifold.knn_simple(samp, 10)
    dist2 = manifold.knn(samp, 10)
    numpy.testing.assert_allclose(dist1.row, dist2.row)
    numpy.testing.assert_allclose(dist1.col, dist2.col)
    numpy.testing.assert_allclose(dist1.data, dist2.data)
    
def test_knn_reduce():
    '''
    '''
    
    samp = numpy.random.rand(32,5)
    
    dist1 = manifold.knn_simple(samp, 20)
    dist2 = manifold.knn(samp, 20)
    
    dist1 = manifold.knn_reduce(dist1, 10)
    dist2 = manifold.knn_reduce(dist2, 10)
    numpy.testing.assert_allclose(dist1.row, dist2.row)
    numpy.testing.assert_allclose(dist1.col, dist2.col)
    numpy.testing.assert_allclose(dist1.data, dist2.data)
    cnt = numpy.zeros(32)
    for i in xrange(dist1.data.shape[0]):
        cnt[dist1.row[i]] += 1
    numpy.testing.assert_(numpy.alltrue(cnt==22), "Unaligned row vector: %d -- %d -- %s"%(numpy.min(cnt), numpy.max(cnt), str(numpy.argwhere(cnt!=22))))

def test_knn_reduce_mutual():
    '''
    '''
    
    samp = numpy.random.rand(32,5)
    
    dist1 = manifold.knn_simple(samp, 20)
    dist2 = manifold.knn(samp, 20)
    
    dist1 = manifold.knn_reduce(dist1, 10, True)
    dist2 = manifold.knn_reduce(dist2, 10, True)
    numpy.testing.assert_allclose(dist1.row, dist2.row)
    numpy.testing.assert_allclose(dist1.col, dist2.col)
    numpy.testing.assert_allclose(dist1.data, dist2.data)

def test_knn_reduce_mutual_explicit():
    '''
    '''
    
    samp = numpy.asarray([ 10, 20, #0
                           10, 19, #1
                           10, 18, #2 
                           10, 17, #3
                           10, 16  #4
                          ]).reshape(5, 2)
    
    row = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    col = [0, 1, 2, 3, 1, 0, 2, 3, 2, 1, 3, 0, 3, 2, 4, 1, 4, 3, 2, 1]
    dist2 = manifold.knn(samp, 3)
    
    numpy.testing.assert_allclose(row, dist2.row)
    numpy.testing.assert_allclose(col, dist2.col)
    
    
    row = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,   0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    col = [0, 1, 2, 1, 0, 2, 2, 1, 3, 3, 2, 4, 4, 3, 2,   0, 1, 2, 1, 0, 2, 2, 1, 3, 3, 2, 4, 4, 3, 2]
    dist2 = manifold.knn_reduce(dist2, 2, False)
    numpy.testing.assert_allclose(row, dist2.row)
    numpy.testing.assert_allclose(col, dist2.col)
    
    dist2 = manifold.knn(samp, 3)
    assert_sorted(dist2)
    row = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    col = [0, 1, 2, 1, 0, 2, 3, 2, 1, 3, 0, 3, 2, 4, 1, 4, 3]
    dist2 = manifold.knn_reduce(dist2, 3, True)
    numpy.testing.assert_allclose(row, dist2.row)
    numpy.testing.assert_allclose(col, dist2.col)

def assert_sorted(dist):
    '''
    '''
    
    if scipy.sparse.issparse(dist):
        assert(numpy.all(dist.row[:-2] <= dist.row[1:-1]))
    else:
        assert(numpy.all(dist[:-2] <= dist[1:-1]))

