''' Defines a set of utility functions to deal with namedtuples



.. Created on Nov 22, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import format_utility
import numpy

def tuple2numpy(blobs, out=None, convert=None):
    ''' Convert a list of namedtuples to a numpy array
    
    >>> from arachnid.core.metadata.format_utility import *
    >>> tuple2numpy([NTuple(id=1, select=2), NTuple(id=5, select=0)])
    array([1,2],[5,0])
    
    .. todo:: make rlnImageName or id a variable?
    
    
    :Parameters:
    
    blobs : list
            List of namedtuples
    out : array, optional
          Array of the correct size
    convert : functor, optional
              Convert a value
    
    :Returns:
    
    out : array
          Array containing all the values of the input
    header : list
             List of headers for each column of the array
    '''
    
    if len(blobs) == 0: return []
    header = list(blobs[0]._fields)
    idx=None
    if format_utility.has_file_id(blobs):
        header.insert(0, "fileid")
        if out is None: out = numpy.zeros((len(blobs), len(blobs[0])+1))
        if hasattr(blobs[0], 'id'):
            for i in xrange(len(blobs)):
                out[i, 0] = format_utility.file_id(blobs[i].id)
                out[i, 1:] = numpy.asarray(blobs[i]._replace(id=format_utility.object_id(blobs[i].id)))
        elif hasattr(blobs[0], 'rlnImageName'):
            for i in xrange(len(blobs)):
                if convert is not None:
                    fid, pid = convert(blobs[i].rlnImageName)
                    tmp = numpy.asarray(blobs[i]._replace(rlnImageName=pid))
                else:
                    fid, tmp = format_utility.file_id(blobs[i].rlnImageName), numpy.asarray(blobs[i]._replace(id=format_utility.object_id(blobs[i].rlnImageName)))
                out[i, 0] = fid
                if idx is None:
                    try:
                        out[i, 1:] = tmp
                    except:
                        tmp = blobs[i]._replace(rlnImageName=pid)
                        idx = []
                        for i in xrange(len(tmp)):
                            try: float(tmp[i])
                            except: pass
                            else: idx.append(i)
                if idx is not None:
                    tmp = blobs[i]._replace(rlnImageName=pid)
                    out[i, 1:1+len(idx)] = numpy.asarray([float(tmp[j]) for j in idx])
                    
        else: raise ValueError, "Cannot find ID column"
    else:
        if out is None: out = numpy.zeros((len(blobs), len(blobs[0])))
        for i in xrange(len(blobs)):
            out[i, :] = numpy.asarray(blobs[i])
    if idx is not None:
        headerold=list(header)
        header=[]
        for i in idx:
            header.append(headerold[i])
    return out, header


