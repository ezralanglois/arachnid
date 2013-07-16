''' Add negative set back and remove random good set

Download to edit and run: :download:`random_good_removal.py <../../arachnid/snippets/random_good_removal.py>`

To run:

.. sourcecode:: sh
    
    $ python random_good_removal.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/random_good_removal.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
from arachnid.core.metadata import format
from arachnid.core.image import ndimage_file
import numpy

if __name__ == '__main__':

    # Parameters
    stack_file = sys.argv[1]
    select_file = sys.argv[2]
    output = sys.argv[3]
    
    total = ndimage_file.count_images(stack_file)
    select, header = format.read(select_file, ndarray=True)
    
    select = select[:, 0].astype(numpy.int)
    numpy.random.shuffle(select)
    bad = select[:total-select.shape[0]]-1
    good = numpy.ones(total, dtype=numpy.bool)
    good[bad]=0
    select = numpy.argwhere(good).squeeze()+1
    select = numpy.vstack((select, numpy.ones(len(select)))).T
    format.write(output, select, header=header)