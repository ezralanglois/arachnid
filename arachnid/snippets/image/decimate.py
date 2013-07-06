''' Decimate images in a stack

Download to edit and run: :download:`decimate.py <../../arachnid/snippets/decimate.py>`

To run:

.. sourcecode:: sh
    
    $ python decimate.py

.. literalinclude:: ../../arachnid/snippets/decimate.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')


#from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file, eman2_utility
#import numpy

if __name__ == '__main__':

    # Parameters
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    bin_factor = int(sys.argv[3])
    
    for i, img in enumerate(ndimage_file.iter_images(input_file)):
        img = eman2_utility.decimate(img, bin_factor)
        ndimage_file.write_image(output_file, img, i)





