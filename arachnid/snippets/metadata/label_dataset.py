''' This script adds a select column to a dataset

Download to edit and run: :download:`label_dataset.py <../../arachnid/snippets/label_dataset.py>`

To run:

.. sourcecode:: sh
    
    $ python label_dataset.py

.. literalinclude:: ../../arachnid/snippets/label_dataset.py
   :language: python
   :lines: 17-
   :linenos:
'''
import sys, os
sys.path.append(os.path.expanduser('~/workspace/arachnida/src'))
from arachnid.core.metadata import format, format_utility
import numpy

if __name__ == '__main__':

    # Parameters
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    good_range = int(sys.argv[3])
    
    
    # Read an alignment file
    data, header = format_utility.tuple2numpy(format.read(input_file, numeric=True))
    if 'select' not in set(header):
        data2 = numpy.hstack((data, numpy.zeros((data.shape[0], 1))))
        data2[:good_range, data.shape[1]] = 1
        format.write(output_file, data2, header=header+['select'])
    else:
        data[:good_range, header.index('select')] = 1
        data[good_range:, header.index('select')] = 0
        format.write(output_file, data, header=header)