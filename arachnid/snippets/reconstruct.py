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
from arachnid.core.image import ndimage_file, eman2_utility
from arachnid.core.image import reconstruct
import itertools 

if __name__ == "__main__":
    align_file = ""
    image_file = ""
    output = ""
    use_rtsq = False
    
    #Note: this code assuming you are reconstructing a dala stack or a translated stack
    
    # Read an alignment file
    align = format.read_alignment(align_file)
    align,header = format_utility.tuple2numpy(align)
    
    if 1 == 0:
        # Create a list ids from 1 to n
        image_ids = xrange(1, len(align)+1) 
        # Create a list of SPIDER filenames from the id list - memory efficient
        iter_single_image_files = itertools.imap(lambda id: spider_utility.spider_filename(image_file, id), image_ids)
        # Read in a set of images from single SPIDER files - memory efficient
        iter_single_images = itertools.imap(ndimage_file.read_image, iter_single_image_files)
        if use_rtsq:
            iter_single_images = itertools.imap(eman2_utility.rot_shift2D, iter_single_images, align)
        # Peform reconstruction
        vol = reconstruct.reconstruct_nn4(iter_single_images, align)
        # Write volume to file
        ndimage_file.write_image(output, vol)
    else:
        # Individual stacks with a pySPIDER alignment file
        align[:, 17]-=1
        iter_single_images = ndimage_file.iter_images(image_file, align[:, 16:18])
        # Peform reconstruction
        vol = reconstruct.reconstruct_nn4(iter_single_images, align)
        # Write volume to file
        ndimage_file.write_image(output, vol)
    
    
    