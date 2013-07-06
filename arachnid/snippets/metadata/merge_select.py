''' Merge the output from montagefromdoc and ctf-select

Download to edit and run: :download:`merge_select.py <../../arachnid/snippets/merge_select.py>`

To run:

.. sourcecode:: sh
    
    $ python merge_select.py

.. literalinclude:: ../../arachnid/snippets/merge_select.py
   :language: python
   :lines: 16-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility
import numpy


if __name__ == '__main__':

    # Parameters
    
    montage_select_file = ""
    ctf_select_file = ""
    output_file = ""
    
    micrographs, header = format_utility.tuple2numpy(format.read(ctf_select_file, numeric=True))
    
    selection = numpy.asarray(format.read(montage_select_file, numeric=True, header='id,select'.split(',')), dtype=numpy.int)[:, 0]-1
    micrographs = micrographs[selection]
    format.write(output_file, micrographs, default_format=format.spiderdoc, header=header)
