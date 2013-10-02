''' Generate a coarse grained stack of view averages

Download to edit and run: :download:`healpix_average.py <../../arachnid/snippets/healpix_average.py>`

    
Right click on link -> "Copy Link Address" or "Copy Link Location"
Download from the command line:

.. sourcecode:: sh
    
    $ wget <paste-url-here>

How to run command:

.. sourcecode:: sh
    
    $ python healpix_average.py "particle_*.spi" bispec_0000.png

.. literalinclude:: ../../arachnid/snippets/healpix_average.py
   :language: python
   :lines: 23-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.metadata import format_alignment #format, 
from arachnid.core.image import ndimage_file, rotate
from arachnid.core.orient import healpix #, orient_utility
import numpy

if __name__ == '__main__':

    # Parameters
    
    image_file = sys.argv[1]        # image_file = "data/dala_01.spi"
    align_file = sys.argv[2]        # align_file = "data/align_01.spi"
    healpix_order=int(sys.argv[3])  # 2
    output_file=sys.argv[4]         # output_file="stack01.spi"
    
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
    
    #count = numpy.zeros(len(ang))
    #n = ndimage_file.count_images(image_file)
    resolution = pow(2, healpix_order)
    for i, img in enumerate(ndimage_file.iter_images(files)):
        ipix = healpix._healpix.ang2pix_ring(resolution, healpix.pmod(numpy.deg2rad(align[i, 1]), numpy.pi), healpix.pmod(numpy.deg2rad(align[i, 2]), 2*numpy.pi))
        #count[ipix]+=1
        rang = rotate.rotate_euler(ang[ipix], align[i, :3])
        rot = -(rang[0]+rang[2])
        #print rot, orient_utility.optimal_inplace_rotation(align[i, :3], ang[ipix].reshape((1, len(ang[ipix]))))[0]
        #assert(rot == orient_utility.optimal_inplace_rotation(align[i, :3], ang[ipix].reshape((1, len(ang[ipix]))))[0])
        #rot = orient_utility.optimal_inplace_rotation(align[i, :3], ang[ipix].reshape((1, len(ang[ipix]))))[0] #-align[i, 0]
        #print "%d of %d -- psi: %f --- ipix: %d"%(i+1, n, rot[0], ipix)
        #sys.stdout.flush()
        avg[ipix] += rotate.rotate_image(img, rot)
    #for i in xrange(len(count)):
    #    print "%d: %d"%(i, count[i])
    ndimage_file.write_stack(output_file, avg)



