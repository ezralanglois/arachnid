''' This script concatenates defocus values on a view classification

Download to edit and run: :download:`concat_defocus.py <../../arachnid/snippets/concat_defocus.py>`

To run:

.. sourcecode:: sh
    
    $ python concat_defocus.py

.. literalinclude:: ../../arachnid/snippets/concat_defocus.py
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
    select_file = sys.argv[3]
    
    
    select, header = format_utility.tuple2numpy(format.read(select_file, numeric=True))
    total = select[:, 2].sum()
    id2defocus = numpy.zeros(total)
    beg=0
    for s in select:
        end = beg + s[2]
        id2defocus[beg:end] = s[3]
        beg=end
    
    view = format.read(input_file, numeric=True)
    for i in xrange(len(view)):
        id = format_utility.split_id(view[i].id, True)[1]
        view[i] = view[i]._replace(c4=id2defocus[id-1])
    format.write(output_file, view)

