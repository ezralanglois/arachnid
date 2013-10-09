''' Generate a coarse grained stack of view averages

Download to edit and run: :download:`healpix_average_data.py <../../arachnid/snippets/healpix_average_data.py>`

    
Right click on link -> "Copy Link Address" or "Copy Link Location"
Download from the command line:

.. sourcecode:: sh
    
    $ wget <paste-url-here>

How to run command:

.. sourcecode:: sh
    
    $ python healpix_average_data.py data_000.spi align.spi 2 view_stack.spi

.. literalinclude:: ../../arachnid/snippets/healpix_average_data.py
   :language: python
   :lines: 23-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.metadata import format_alignment #format, 
from arachnid.core.image import ndimage_file, rotate, eman2_utility
from arachnid.core.orient import healpix, orient_utility
import numpy

if __name__ == '__main__':

    # Parameters
    
    image_file = sys.argv[1]        # image_file = "data/dala_01.spi"
    align_file = sys.argv[2]        # align_file = "data/align_01.spi"
    healpix_order=int(sys.argv[3])  # 2
    output_file=sys.argv[4]         # output_file="stack01.spi"
    mirror = False
    
    ang = healpix.angles(healpix_order)
    files, align = format_alignment.read_alignment(align_file, image_file)
    
    if isinstance(files, list):
        image_file=files[0]
        if isinstance(image_file, tuple):
            image_file=image_file[0]
    img = ndimage_file.read_image(image_file)
    avg = numpy.zeros((len(ang), img.shape[0], img.shape[1]))
    #align = format.read_alignment(align_file, ndarray=True)[0]
    print "image_file: %s with %d projections -> %d views"%(image_file, len(align), len(ang))
    sys.stdout.flush()
    
    align2 = numpy.zeros((align.shape[0], align.shape[1]+1))
    print 'here1', align[:3]
    orient_utility.coarse_angles(healpix_order, align, mirror, align2)
    print 'here2', align2[:3]
    for i, img in enumerate(ndimage_file.iter_images(files)):
        ipix = align2[i, 6]
        rot = align2[i, 3]
        tx = align2[i, 4]
        ty = align2[i, 5]
        img[:] = rotate.rotate_image(img, rot, tx, ty)
        if mirror and align2[i, 1] > 180.0: img[:] = eman2_utility.mirror(img) 
        avg[ipix] += img
    ndimage_file.write_stack(output_file, avg)



