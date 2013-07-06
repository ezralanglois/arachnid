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
        t = align[i, 1]
        if t > 180.0: t -= 180.0
        ipix = healpix._healpix.ang2pix_ring(resolution, numpy.deg2rad(t), numpy.deg2rad(align[i, 2]))
        #count[ipix]+=1
        rang = rotate.rotate_euler(ang[ipix], (-align[i, 3], align[i, 1], align[i, 2]))
        rot = (rang[0]+rang[2])
        rt3d = orient_utility.align_param_2D_to_3D_simple(align[i, 3], align[i, 4], align[i, 5])
        print rt3d
        rot, tx, ty = orient_utility.align_param_2D_to_3D_simple(rot, rt3d[1], rt3d[2])
        print rt3d, '--', rot, tx, ty
        #print rot, orient_utility.optimal_inplace_rotation(align[i, :3], ang[ipix].reshape((1, len(ang[ipix]))))[0]
        #assert(rot == orient_utility.optimal_inplace_rotation(align[i, :3], ang[ipix].reshape((1, len(ang[ipix]))))[0])
        #rot = orient_utility.optimal_inplace_rotation(align[i, :3], ang[ipix].reshape((1, len(ang[ipix]))))[0] #-align[i, 0]
        #print "%d of %d -- psi: %f --- ipix: %d"%(i+1, n, rot[0], ipix)
        #sys.stdout.flush()
        img[:] = rotate.rotate_image(img, rot, tx, ty)
        if align[i, 1] > 180.0: img[:] = eman2_utility.mirror(img)
        avg[ipix] += img
    #for i in xrange(len(count)):
    #    print "%d: %d"%(i, count[i])
    ndimage_file.write_stack(output_file, avg)



