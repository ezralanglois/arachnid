''' Reconstruct a volume from a set of single images

Download to edit and run: :download:`reconstruct_single.py <../../arachnid/snippets/reconstruct_single.py>`

To run:

.. sourcecode:: sh
    
    $ python reconstruct_single.py

.. literalinclude:: ../../arachnid/snippets/reconstruct_single.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
#sys.path.append('/home/robertl/tmp/arachnid-0.0.1/')

from arachnid.core.app import tracing
from arachnid.core.metadata import format, format_utility #, spider_utility
from arachnid.core.image import ndimage_file #, eman2_utility
from arachnid.core.image import reconstruct
import numpy,logging

if __name__ == "__main__":
    image_file = sys.argv[1]
    align_file = sys.argv[2]
    output = sys.argv[3]
    thread_count=16
    
    tracing.configure_logging()
    #Note: this code assuming you are reconstructing a dala stack or a translated stack
    
    # Read an alignment file
    
    align = format.read_alignment(align_file)
    logging.error("Reconstructing %d particles"%len(align))
    align,header = format_utility.tuple2numpy(align)
    assert(header[0]=='id')
    index = align[:, 0].astype(numpy.int)
    align[:, 1] = numpy.rad2deg(align[:, 1])
    iter_single_images = ndimage_file.iter_images(image_file, index)
    image_size = ndimage_file.read_image(image_file).shape[0]
    vol = reconstruct.reconstruct_bp3f_mp(iter_single_images, image_size, align, thread_count=thread_count)
    ndimage_file.write_image(output, vol)
