''' Reconstruct a volume

Download to edit and run: :download:`reconstruct3.py <../../arachnid/snippets/reconstruct3.py>`

To run:

.. sourcecode:: sh
    
    $ python reconstruct3.py

.. literalinclude:: ../../arachnid/snippets/reconstruct3.py
   :language: python
   :lines: 16-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file
from arachnid.core.image import reconstruct
import itertools, numpy


if __name__ == "__main__":
    align_file = ""
    image_file = ""
    output = ""
    
    #Note: this code assuming you are reconstructing a dala stack or a translated stack
    
    # Read an alignment file
    align = format.read_alignment(align_file)
    align,header = format_utility.tuple2numpy(align)
    
    # Define two subsets - here even and odd
    even = numpy.arange(0, len(align), 2, dtype=numpy.int)
    odd = numpy.arange(1, len(align), 2, dtype=numpy.int)
    # Create a list ids from 1 to n
    image_ids = xrange(1, len(align)+1) 
    # Create a list of SPIDER filenames from the id list - memory efficient
    iter_single_image_files1 = itertools.imap(lambda id: spider_utility.spider_filename(image_file, id), image_ids[even])
    iter_single_image_files2 = itertools.imap(lambda id: spider_utility.spider_filename(image_file, id), image_ids[odd])
    # Read in a set of images from single SPIDER files - memory efficient
    iter_single_images1 = itertools.imap(ndimage_file.read_image, iter_single_image_files1)
    iter_single_images2 = itertools.imap(ndimage_file.read_image, iter_single_image_files2)
    # Peform reconstruction
    vol,vol_even,vol_odd = reconstruct.reconstruct_nn4_3(iter_single_images1, iter_single_images2, align[even], align[odd])
    # Write volume to file
    ndimage_file.write_image(output, vol)
    ndimage_file.write_image(format_utility.add_prefix(output, "even_"), vol_even)
    ndimage_file.write_image(format_utility.add_prefix(output, "odd_"), vol_odd)
    
    
    