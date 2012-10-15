''' Defines routines missing in older versions of NumPy

.. Created on Oct 15, 2012
.. codeauthor:: robertlanglois
'''
import numpy

if not hasattr(numpy.random, 'choice'):
    def choice(a, size=1, replace=True, p=None):
            """
            choice(a, size=1, replace=True, p=None)
    
            Generates a random sample from a given 1-D array
    
                    .. versionadded:: 1.7.0
    
            Parameters
            -----------
            a : 1-D array-like or int
                If an ndarray, a random sample is generated from its elements.
                If an int, the random sample is generated as if a was numpy.arange(n)
            size : int
                Positive integer, the size of the sample.
            replace : boolean, optional
                Whether the sample is with or without replacement
            p : 1-D array-like, optional
                The probabilities associated with each entry in a.
                If not given the sample assumes a uniform distribtion over all
                entries in a.
    
            Returns
            --------
            samples : 1-D ndarray, shape (size,)
                The generated random samples
    
            Raises
            -------
            ValueError
                If a is an int and less than zero, if a or p are not 1-dimensional,
                if a is an array-like of size 0, if p is not a vector of
                probabilities, if a and p have different lengths, or if
                replace=False and the sample size is greater than the population
                size
    
            See Also
            ---------
            randint, shuffle, permutation
    
            Examples
            ---------
            Generate a uniform random sample from numpy.arange(5) of size 3:
    
            >>> np.random.choice(5, 3)
            array([0, 3, 4])
            >>> #This is equivalent to np.random.randint(0,5,3)
    
            Generate a non-uniform random sample from np.arange(5) of size 3:
    
            >>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
            array([3, 3, 0])
    
            Generate a uniform random sample from np.arange(5) of size 3 without
            replacement:
    
            >>> np.random.choice(5, 3, replace=False)
            array([3,1,0])
            >>> #This is equivalent to np.random.shuffle(np.arange(5))[:3]
    
            Generate a non-uniform random sample from np.arange(5) of size
            3 without replacement:
    
            >>> np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
            array([2, 3, 0])
    
            Any of the above can be repeated with an arbitrary array-like
            instead of just integers. For instance:
    
            >>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
            >>> np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
            array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'],
                  dtype='|S11')
    
            """
    
            # Format and Verify input
            if isinstance(a, int):
                if a > 0:
                    pop_size = a #population size
                else:
                    raise ValueError("a must be greater than 0")
            else:
                a = numpy.array(a, ndmin=1, copy=0)
                if a.ndim != 1:
                    raise ValueError("a must be 1-dimensional")
                pop_size = a.size
                if pop_size is 0:
                    raise ValueError("a must be non-empty")
    
            if None != p:
                p = numpy.array(p, dtype=numpy.double, ndmin=1, copy=0)
                if p.ndim != 1:
                    raise ValueError("p must be 1-dimensional")
                if p.size != pop_size:
                    raise ValueError("a and p must have same size")
                if numpy.any(p < 0):
                    raise ValueError("probabilities are not non-negative")
                if not numpy.allclose(p.sum(), 1):
                    #_logger.error("dP=%f"%(1.0-p.sum()))
                    raise ValueError("probabilities do not sum to 1")
    
            # Actual sampling
            if replace:
                if None != p:
                    cdf = p.cumsum()
                    cdf /= cdf[-1]
                    uniform_samples = numpy.random.random(size)
                    idx = cdf.searchsorted(uniform_samples, side='right')
                else:
                    idx = numpy.random.randint(0, pop_size, size=size)
            else:
                if size > pop_size:
                    raise ValueError(''.join(["Cannot take a larger sample than ",
                                              "population when 'replace=False'"]))
    
                if None != p:
                    if numpy.sum(p > 0) < size:
                        raise ValueError("Fewer non-zero entries in p than size")
                    n_uniq = 0
                    p = p.copy()
                    found = numpy.zeros(size, dtype=numpy.int)
                    while n_uniq < size:
                        x = numpy.random.rand(size - n_uniq)
                        if n_uniq > 0:
                            p[found[0:n_uniq]] = 0
                        cdf = numpy.cumsum(p)
                        cdf /= cdf[-1]
                        new = cdf.searchsorted(x, side='right')
                        new = numpy.unique(new)
                        found[n_uniq:n_uniq + new.size] = new
                        n_uniq += new.size
                    idx = found
                else:
                    idx = numpy.random.permutation(pop_size)[:size]
    
            #Use samples as indices for a if a is array-like
            if isinstance(a, int):
                return idx
            else:
                return a.take(idx)
else: choice = numpy.random.choice

