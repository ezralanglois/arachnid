''' Pad a dataset with projections extracted from micrograph based on cross-correlation

Download to edit and run: :download:`pad_dataset.py <../../arachnid/snippets/pad_dataset.py>`

To run:

.. sourcecode:: sh
    
    $ python pad_dataset.py

.. literalinclude:: ../../arachnid/snippets/pad_dataset.py
   :language: python
   :lines: 16-
   :linenos:
'''
#import sys
#sys.path.append('/home/robertl/tmp/arachnid-0.0.1/')
from arachnid.core.metadata import format
from arachnid.core.image import ndimage_file, ndimage_utility
import logging


if __name__ == '__main__':

    # Parameters
    
    micrograph_file = ""
    stack_file = ""
    stack_bench = ""
    coord_file = ""
    output_file = ""
    bin_factor = 1.0
    
    logging.basicConfig()
    
    total = ndimage_file.count_images(stack_bench)
    if ndimage_file.count_images(stack_file) < total:
        offset = 0
        for img in ndimage_file.iter_images(stack_file):
            ndimage_file.write_image(output_file, img, offset)
            offset +=1
        
        coords, header = format.read(coord_file, ndarray=True)
        coords = coords[:, (header.index('x'), header.index('y'))]
        coords = coords[::-1]
        coords = coords[:(total-offset)]
        
        npmic = ndimage_file.read_image(micrograph_file)
        width = img.shape[0]
        for win in ndimage_utility.for_each_window(npmic, coords, width, bin_factor):
            ndimage_file.write_image(output_file, win, offset)
            offset +=1
    else:
        logging.warn("Skipping: %s"%stack_file)