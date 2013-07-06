''' Calculate the radon transform of an image

Download to edit and run: :download:`radon_image.py <../../arachnid/snippets/radon_image.py>`

To run:

.. sourcecode:: sh
    
    $ python radon_image.py

.. literalinclude:: ../../arachnid/snippets/radon_image.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')


from arachnid.core.image import ndimage_file, ndimage_utility
import numpy

if __name__ == '__main__':

    # Parameters
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    img=ndimage_file.read_image(input_file)
    
    rimg=ndimage_utility.normalize_min_max(img, 0, 255)
    rimg = rimg.astype(numpy.int32)
    rimg = ndimage_utility.frt2(rimg)
    ndimage_file.write_image(output_file, rimg)
    
    
