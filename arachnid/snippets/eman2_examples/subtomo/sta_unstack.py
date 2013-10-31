''' Unstack a volume from a stack of volumes

Download to edit and run: :download:`sta_unstack.py <../../arachnid/snippets/sta_unstack.py>`

To run:

.. sourcecode:: sh
    
    $ python sta_unstack.py

.. note::
    
    Requires EMAN2 2.1

.. literalinclude:: ../../arachnid/snippets/sta_unstack.py
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
    subtomo_stack = sys.argv[1]
    output = sys.argv[2]
    index = int(sys.argv[3])
    
    e = EMAN2.EMData()
    e.read_image(subtomo_stack, index)
    e.process_inplace('filter.lowpass.gauss', dict(cutoff_freq=0.025))
    e.write_image(output)

