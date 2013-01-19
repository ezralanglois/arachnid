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
from arachnid.core.metadata import format, format_utility
from arachnid.core.image import ndimage_file
from arachnid.core.image import reconstruct
from arachnid.core.parallel import mpi_utility
import logging, numpy

if __name__ == "__main__":
    
    tracing.configure_logging()
    param={}
    mpi_utility.mpi_init(param, use_MPI=True)
    
    
    image_file = sys.argv[1]   # phase_flip_dala_stack.spi
    align_file = sys.argv[2]   # align.spi
    output = sys.argv[3]       # raw_vol.spi
    node_tmp_file=sys.argv[4]  # /tmp_data/username/phase_flip_dala_part.spi
    thread_count = 8
    type='bp3f' # bp3f or nn4
    
    # Read an alignment file
    align = format.read_alignment(align_file)
    logging.error("Reconstructing %d particles"%len(align))
    align,header = format_utility.tuple2numpy(align)
    selection = numpy.arange(len(align), dtype=numpy.int)
    
    # Split data
    curr_slice = mpi_utility.mpi_slice(len(align), **param)
    
    # Cache split windows locally
    image_file=ndimage_file.copy_local(image_file, selection[curr_slice], node_tmp_file, **param)
    
    
    selection = numpy.arange(len(selection[curr_slice]), dtype=numpy.int)
    iter_single_images = ndimage_file.iter_images(image_file, selection)
    image_size = ndimage_file.read_image(image_file).shape[0]
    if type=='bp3f':
        vol = reconstruct.reconstruct_bp3f_mp(iter_single_images, image_size, align[curr_slice], thread_count=thread_count, **param)
    else:
        vol = reconstruct.reconstruct_nn4f_mp(iter_single_images, image_size, align[curr_slice], thread_count=thread_count, **param)
    if vol is not None: 
        ndimage_file.write_image(output, vol)
