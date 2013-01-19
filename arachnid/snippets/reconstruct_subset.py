''' Reconstruct a volume from a subset of projections

Download to edit and run: :download:`reconstruct_subset.py <../../arachnid/snippets/reconstruct_subset.py>`

To run:

.. sourcecode:: sh
    
    $ python reconstruct_subset.py

.. literalinclude:: ../../arachnid/snippets/reconstruct_subset.py
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
    select_file = sys.argv[3]
    output = sys.argv[4]
    use_rtsq = False
    thread_count=16
    mode=2
    
    tracing.configure_logging()
    #Note: this code assuming you are reconstructing a dala stack or a translated stack
    
    # Read an alignment file
    
    align = format.read_alignment(align_file)
    logging.error("Reconstructing %d particles"%len(align))
    align,header = format_utility.tuple2numpy(align)
    selectidx = format_utility.tuple2numpy(format.read_alignment(select_file))[0][:, 0].astype(numpy.int)
    
    if mode == 0:
        index = numpy.arange(len(align), dtype=numpy.int)
    elif mode == 1:
        index = selectidx
    elif mode == 2:
        index = numpy.arange(len(align), dtype=numpy.int)
        numpy.random.shuffle(index)
        index = index[:selectidx.shape[0]]
    elif mode==3:
        neg = align.shape[0]-selectidx.shape[0]
        numpy.random.shuffle(selectidx)
        index = selectidx[:neg]
    else:
        select = numpy.zeros(len(align), dtype=numpy.bool)
        select[selectidx]=1
        select=numpy.logical_not(select)
        index = numpy.argwhere(select).squeeze()
    
    align = align[index]
    iter_single_images = ndimage_file.iter_images(image_file, index)
    if thread_count < 2:
        vol = reconstruct.reconstruct_nn4(iter_single_images, align)
    else:
        image_size = ndimage_file.read_image(image_file).shape[0]
        vol = reconstruct.reconstruct_nn4_mp(iter_single_images, image_size, align, thread_count=thread_count)
    ndimage_file.write_image(output, vol)
