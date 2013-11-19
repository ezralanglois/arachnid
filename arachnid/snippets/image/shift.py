''' This centers particles in the window

Download to edit and run: :download:`shif.py <../../arachnid/snippets/shift.py>`

To run:

.. sourcecode:: sh
    
    $ python shif.py

.. literalinclude:: ../../arachnid/snippets/shift.py
   :language: python
   :lines: 17-
   :linenos:
'''
import sys
from arachnid.core.metadata import format, spider_utility, relion_utility
from arachnid.core.image import ndimage_file, ndimage_utility

if __name__ == '__main__':

    # Parameters
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    mult=1.0
    
    
    # Read an alignment file
    align = format.read(input_file, numeric=True)
    
    for i in xrange(len(align)):
        filename, id = relion_utility.relion_file(align[i].rlnImageName)
        img = ndimage_file.read_image(filename, id-1)
        img = ndimage_utility.fourier_shift(img, align[i].rlnOriginX, align[i].rlnOriginY)
        ndimage_file.write_image(spider_utility.spider_filename(output_file, filename), img, id-1)
    