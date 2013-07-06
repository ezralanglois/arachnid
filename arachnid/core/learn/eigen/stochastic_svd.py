''' Compute parallel eigen decompostion using a one-pass parallel approximation

Original Code is from https://github.com/t-brandt/acorns-adi/blob/master/svd/stochastic_svd.py
Original Algorithm is presented in Halko, Martinsson, & Tropp, arXiv:0909.4061

.. Created on Mar 14, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import numpy, scipy
from arachnid.core import process_tasks

def matrix_multiply_AbA(A, b):
    ''' Mulitply sample matrix A by 
    '''
    
    return numpy.dot(A.T, numpy.dot(A, b))

def randmatmult(A, n):
    ''' Draw a random test matrix omega from the sample matrix A
    '''
    
    o = numpy.random.normal(0., 1., (A.shape[0], n))
    return numpy.dot(A.T, o)

def gram_schmidt(A):
    ''' QR factorization using Gram-Schmidt to orthonormalize a full column rank matrix A
    
    :Parameters:
    
    A : array
        Matrix to compute factorization on
    
    :Returns:
    
    out : array
          QR factorization of A
    '''
    
    V = A.T
    # Gram-Schmidt
    # Normalize each column, then subtract its components from other columns
    for i in xrange(V.shape[0]):
        V[i] /= numpy.sqrt(numpy.sum(V[i]**2))
        for j in xrange(i + 1, V.shape[0]):
            V[j] -= V[i] * numpy.sum(V[i] * V[j])
    return V.T

def stochastic_svd(A, rank=20, extra_dims=None, power_iter=2, chunksize=50, thread_count=1):
    ''' Stochastic singular value decomposition (SV)
    
    :Parameters:
    
    
    '''
    
    A = numpy.asarray(A)
    if A.ndim != 2: "Requires 2-dimensional array as input"
    samples = max(10, 2 * rank) if extra_dims is None else rank + int(extra_dims)    
    y = numpy.zeros((A.shape[1], samples))
    
    # Generate random matrix
    def chunk_array(A, val):
        for i1 in xrange(0, A.shape[0], chunksize):
            i2 = min(i1 + chunksize, A.shape[0])
            yield A[i1:i2], val
        
    for val in process_tasks.iterate_map(chunk_array(A, samples), randmatmult, thread_count):
        y += val
    y = gram_schmidt(y)
    
    yold = y.copy()
    for iter in xrange(power_iter):
        yold[:] = y
        y[:]=0
        for val in process_tasks.iterate_map(chunk_array(A, yold), matrix_multiply_AbA, thread_count):
            y += val
        y = gram_schmidt(y)
    del yold
    
    
    y = y.T
    b = numpy.zeros((y.shape[0], A.shape[0]))
    
    def chunk_array2(A, val):
        for i1 in xrange(0, A.shape[0], chunksize):
            i2 = min(i1 + chunksize, A.shape[0])
            yield i1, i2, val, A[i1:i2].T
    def dot_index(i1, i2, v1, v2):
        return i1, i2, numpy.dot(v1, v2)
    for i1, i2, val in process_tasks.iterate_map(chunk_array2(A, y), dot_index, thread_count):
        b[:, i1:i2]=val
        
    
    u, v, s = scipy.linalg.svd(b)
    u = numpy.dot(y.T, u)
    return u
