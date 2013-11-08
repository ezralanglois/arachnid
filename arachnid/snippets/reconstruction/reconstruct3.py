''' Reconstruct a volume from a subset of projections

Download to edit and run: :download:`reconstruct3.py <../../arachnid/snippets/reconstruct3.py>`

To run:

.. sourcecode:: sh
    
    $ python reconstruct3.py

.. literalinclude:: ../../arachnid/snippets/reconstruct3.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
from arachnid.core.app import tracing
from arachnid.core.metadata import format, format_utility, spider_params
from arachnid.core.image import ndimage_file, ctf, rotate
from arachnid.core.image import reconstruct
import logging, numpy

def process_image(img, data, **extra):
    '''
    '''
    
    img = rotate.rotate_image(img, data[5], data[6], data[7])
    ctfimg = ctf.phase_flip_transfer_function(img.shape, data[17], **extra)
    img = ctf.correct(img, ctfimg)
    return img

if __name__ == "__main__":
    
    tracing.configure_logging()

    image_file = sys.argv[1]   # phase_flip_dala_stack.spi
    align_file = sys.argv[2]   # align.spi
    param_file = sys.argv[3]
    output = sys.argv[4]       # raw_vol.spi
    bin_factor = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0        # raw_vol.spi
    thread_count = 32
    
    type='bp3f' # bp3f or nn4
    extra = spider_params.read(param_file)
    extra.update(spider_params.update_params(bin_factor, **extra))
    print "Loaded param file"
    extra.update(thread_count=thread_count)
    
    align = format.read_alignment(align_file)
    logging.error("Reconstructing %d particles"%len(align))
    align,header = format_utility.tuple2numpy(align)
    if align.shape[1] > 17:
        selection = align[:, 15:17]
        selection[:, 1]-=1
        #align[:, 6:8] /= extra['apix']
    else:
        selection = align[:, 4].astype(numpy.int)-1
    
    image_size = ndimage_file.read_image(image_file).shape[0]
    even = numpy.arange(0, len(selection), 2, dtype=numpy.int)
    odd = numpy.arange(1, len(selection), 2, dtype=numpy.int)
    iter_single_images1 = ndimage_file.iter_images(image_file, selection[even])
    iter_single_images2 = ndimage_file.iter_images(image_file, selection[odd])
    align1 = align[even]
    align2 = align[odd]
    vol = reconstruct.reconstruct3_bp3f_mp(image_size, iter_single_images1, iter_single_images2, align1, align2, **extra)
    if vol is not None: 
        ndimage_file.write_image(output, vol[0])
        ndimage_file.write_image(format_utility.add_prefix(output, 'h1_'), vol[1])
        ndimage_file.write_image(format_utility.add_prefix(output, 'h2_'), vol[2])



