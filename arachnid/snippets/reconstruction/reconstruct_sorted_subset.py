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
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.app import tracing
from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file
from arachnid.core.image import reconstruct
import numpy,logging

if __name__ == "__main__":
    image_file = sys.argv[1]
    align_file = sys.argv[2]
    select_file = sys.argv[3]
    col = int(sys.argv[4])
    output = sys.argv[5]
    bin_size=20000
    use_rtsq = False
    thread_count=16
    mode=0
    #python reconstruct_sort.py data/dala05.ter data/align_05.ter output/sel_r9_01.ter 4 vol_order_000.spi
    
    tracing.configure_logging(log_level=3)
    #Note: this code assuming you are reconstructing a dala stack or a translated stack
    
    # Read an alignment file
    
    align1,header = format.read_alignment(align_file, ndarray=True)
    logging.error("Reconstructing %d particles"%len(align1))
    order = format.read(select_file, ndarray=True)[0][:, col]
    sindex = numpy.argsort(order)
    tot = len(sindex)/bin_size
    logging.info("Column %d: %f"%(col, order[0]))
    
    b=0
    for i in xrange(tot):
        e = b+ bin_size
        logging.info("Iteration: %d of %d reconstructing %d-%d (%d)"%(i+1, tot, b, e, (e-b)))
        index = sindex[b:e]
        align = align1[index]
        iter_single_images = ndimage_file.iter_images(image_file, index)
        if thread_count < 2:
            vol = reconstruct.reconstruct_bp3f_mp(iter_single_images, align)
        else:
            image_size = ndimage_file.read_image(image_file).shape[0]
            vol = reconstruct.reconstruct_bp3f_mp(iter_single_images, image_size, align, thread_count=thread_count)
        ndimage_file.write_image(spider_utility.spider_filename(output, i+1), vol)
        b=e

