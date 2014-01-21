''' Select one class from PCA data

Download to edit and run: :download:`unary_classify.py <../../arachnid/snippets/unary_classify.py>`

To run:

.. sourcecode:: sh
    
    $ python unary_classify.py

.. literalinclude:: ../../arachnid/snippets/unary_classify.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
from arachnid.core.metadata import format
from arachnid.core.learn import unary_classification
import numpy

if __name__ == '__main__':

    # Parameters
    
    input = sys.argv[1]
    output = sys.argv[2]
    
    data, header = format.read(input, ndarray=True)
    c_1 = header.index('c_1')
    c_2 = header.index('c_2')
    data2 = data[:, (c_1, c_2)]
    
    if 1 == 1:
        from sklearn.cluster import MeanShift, estimate_bandwidth
        bandwidth = estimate_bandwidth(data2, quantile=0.01, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(data2)
        
        labels = numpy.unique(ms.labels_)
        print len(labels)
        largest = (0, 0)
        for i in xrange(len(labels)):
            tot = numpy.sum(labels[i]==ms.labels_)
            print labels[i], tot
            if tot > largest[0]: largest = (tot, labels[i])
        data2 = numpy.zeros((data.shape[0], data.shape[1]+1))
        data2[:, 0]=data[:, 0]
        data2[:, 1] = ms.labels_
        data2[:, 2:]=data[:, 1:]
        format.write(output, data2, header=[header[0], 'select']+header[1:])
        
        index = numpy.argwhere(ms.labels_==largest[1]).ravel()+1
        non_index = numpy.argwhere(numpy.logical_not(ms.labels_==largest[1])).ravel()+1
        format.write(output, numpy.hstack((index[:, numpy.newaxis], numpy.ones(len(index))[:, numpy.newaxis])), header=['id', 'select'], prefix="sel_")
        format.write(output, numpy.hstack((non_index[:, numpy.newaxis], numpy.ones(len(non_index))[:, numpy.newaxis])), header=['id', 'select'], prefix="non_")
        sys.exit(0)
    
    print c_1, c_2
    print data.shape
    
    
    
    #sel = unary_classification.mahalanobis_with_chi2(data2, 0.1)
    sel = unary_classification.robust_euclidean(data2, 0.001)
    print numpy.sum(sel)
    #sel2 = unary_classification.mahalanobis_with_chi2(data2[sel], 0.4)
    #sel[sel][numpy.logical_not(sel2)]=0
    #print numpy.sum(sel)
    index = numpy.argwhere(sel).ravel()+1
    non_index = numpy.argwhere(numpy.logical_not(sel)).ravel()+1
    if 1 == 0:
        format.write(output, numpy.hstack((index[:, numpy.newaxis], numpy.ones(len(index))[:, numpy.newaxis])), header=['id', 'select'])
        format.write(output, numpy.hstack((non_index[:, numpy.newaxis], numpy.ones(len(non_index))[:, numpy.newaxis])), header=['id', 'select'], prefix="non_")
    else:
        data2 = numpy.zeros((data.shape[0], data.shape[1]+1))
        data2[:, 0]=data[:, 0]
        data2[:, 1] = sel
        data2[:, 2:]=data[:, 1:]
        format.write(output, data2, header=[header[0], 'select']+header[1:])
        
    
    