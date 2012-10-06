''' Reconstruct a volume

Download to edit and run: :download:`reconstruct.py <../../arachnid/snippets/reconstruct.py>`

To run:

.. sourcecode:: sh
    
    $ python reconstruct.py

.. literalinclude:: ../../arachnid/snippets/reconstruct.py
   :language: python
   :lines: 16-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file
from arachnid.core.image import reconstruct
import itertools 

if __name__ == "__main__":
    align_file = ""
    image_file = ""
    output = ""
    
    #Note: this code assuming you are reconstructing a dala stack or a translated stack
    
    # Read an alignment file
    align = format.read_alignment(align_file)
    align,header = format_utility.tuple2numpy(align)
    
    # Create a list ids from 1 to n
    image_ids = xrange(1, len(align)+1) 
    # Create a list of SPIDER filenames from the id list - memory efficient
    iter_single_image_files = itertools.imap(lambda id: spider_utility.spider_filename(image_file, id), image_ids)
    # Read in a set of images from single SPIDER files - memory efficient
    iter_single_images = itertools.imap(ndimage_file.read_image, iter_single_image_files)
    # Peform reconstruction
    vol = reconstruct.reconstruct_nn4(iter_single_images, align)
    # Write volume to file
    ndimage_file.write_image(output, vol)
    
    
    