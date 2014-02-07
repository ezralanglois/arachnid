''' Unstacks an image stack into individual images

The output files are named according to the SPIDER convention: filename_00001.dat.

Download to edit and run: :download:`unstack.py <../../arachnid/snippets/image/unstack.py>`

To run:

.. sourcecode:: sh
    
    $ python unstack.py

.. note::
    
    You must have Arachnid installed to run this script

.. literalinclude:: ../../arachnid/snippets/image/unstack.py
   :language: python
   :lines: 20-
   :linenos:
'''
from arachnid.core.metadata import spider_utility
from arachnid.core.image import ndimage_file

if __name__ == '__main__':
    
    # Parameters
    
    input_stack = "stack_001.dat"
    output_image = 'image_001_00000.dat'
    
    # Script
    
    
    
    for i, img in enumerate(ndimage_file.iter_images(input_stack)):
        output_image = spider_utility.spider_filename(output_image, i+1)
        ndimage_file.write_image(output_image, img)
