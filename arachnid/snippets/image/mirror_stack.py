''' Mirror images in a stack

Download to edit and run: :download:`mirror_stack.py <../../arachnid/snippets/mirror_stack.py>`

To run:

.. sourcecode:: sh
    
    $ python mirror_stack.py

.. literalinclude:: ../../arachnid/snippets/mirror_stack.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')


from arachnid.core.metadata import format #, format_utility, spider_utility
from arachnid.core.image import ndimage_file, eman2_utility
#import numpy

if __name__ == '__main__':

    # Parameters
    
    input_file = sys.argv[1]
    align_file = sys.argv[2]
    output_file = sys.argv[3]
    bin_factor=2.0
    
    align = format.read_alignment(align_file)
    
    for i, img in enumerate(ndimage_file.iter_images(input_file)):
        if align[i].theta > 179.9:
            img=eman2_utility.mirror(img)
        if bin_factor > 1.0: img = eman2_utility.decimate(img, bin_factor)
        ndimage_file.write_image(output_file, img, i)





