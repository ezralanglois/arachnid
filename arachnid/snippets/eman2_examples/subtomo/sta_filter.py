''' Unstack a volume from a stack of volumes

Download to edit and run: :download:`sta_filter.py <../../arachnid/snippets/sta_filter.py>`

To run:

.. sourcecode:: sh
    
    $ python sta_filter.py

.. note::
    
    Requires EMAN2 2.1

.. literalinclude:: ../../arachnid/snippets/sta_filter.py
   :language: python
   :lines: 20-
   :linenos:
'''
import sys

#from arachnid.core.image.eman2_utility 
import EMAN2
#import numpy

if __name__ == '__main__':

    # Parameters
    input = sys.argv[1]
    output = sys.argv[2]
    
    e = EMAN2.EMData()
    e.read_image(input)
    e.process_inplace('filter.lowpass.gauss', dict(cutoff_freq=0.01))
    e.write_image(output)

