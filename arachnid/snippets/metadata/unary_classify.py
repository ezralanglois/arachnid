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
    print c_1, c_2
    print data[0, (c_1, c_2)]
    sel = unary_classification.mahalanobis_with_chi2(data[:, (c_1, c_2)], 0.97)
    index = numpy.argwhere(sel).ravel()+1
    if 1 == 1:
        format.write(output, numpy.hstack((index[:, numpy.newaxis], numpy.ones(len(index))[:, numpy.newaxis])), header=['id', 'select'])
    else:
        data2 = numpy.zeros((data.shape[0], data.shape[1]+1))
        data2[:, 0]=data[:, 0]
        data2[:, 1] = sel
        data2[:, 2:]=data[:, 1:]
        format.write(output, data2, header=[header[0], 'select']+header[1:])
        
    
    