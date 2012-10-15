''' This script modifies the translations in an alignment file so the 
reconstruction can be done at a different interpolation level.

Download to edit and run: :download:`scale_align.py <../../arachnid/snippets/scale_align.py>`

To run:

.. sourcecode:: sh
    
    $ python scale_align.py

.. literalinclude:: ../../arachnid/snippets/scale_align.py
   :language: python
   :lines: 17-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file, eman2_utility
from arachnid.core.orient import orient_utility
import numpy, itertools

if __name__ == '__main__':

    # Parameters
    
    align_file = ""
    output_file = ""
    mult=2
    
    
    # Read an alignment file
    align = format.read_alignment(align_file)
    align,header = format_utility.tuple2numpy(align)
    
    try:
        tx=header.index('tx')
        ty=header.index('ty')
    except:
        tx=header.index('rlnOriginX')
        ty=header.index('rlnOriginY')
        
    
    align[:, tx] *= mult
    align[:, ty] *= mult
    
    format.write(output_file, align, header=header)