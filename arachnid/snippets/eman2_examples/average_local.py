

''' average_local a volume

Download to edit and run: :download:`average_local.py <../../arachnid/snippets/average_local.py>`

To run:

.. sourcecode:: sh
    
    $ python average_local.py

.. literalinclude:: ../../arachnid/snippets/average_local.py
   :language: python
   :lines: 16-
   :linenos:
'''
#import sys

#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1/')
from arachnid.core.app import tracing
from arachnid.core.metadata import format, format_utility #, spider_utility
from arachnid.core.image import ndimage_file, eman2_utility
from arachnid.core.image import manifold
from arachnid.core.orient import transforms
import numpy, sys, logging
import scipy.ndimage.interpolation

if __name__ == "__main__":
    image_file = sys.argv[1]
    align_file = sys.argv[2]
    output = sys.argv[3]
    use_rtsq = False
    thread_count=16
    nn = 101
    gdist = numpy.deg2rad(15)
    cache_file = "cache_recon"
    
    tracing.configure_logging()
    #Note: this code assuming you are average_localing a dala stack or a translated stack
    
    # Read an alignment file
    align,header = format.read_alignment(align_file, ndarray=True)
    logging.info("Averaging %d particles"%len(align))
    if not eman2_utility.is_avaliable(): raise ValueError, "EMAN2 is not installed"
    reference = numpy.asarray(eman2_utility.utilities.even_angles(15, 0.0, 90.0, 0.0, 359.99, 'P', '', 'c1'))
    ang = numpy.zeros((len(align)+len(reference), 4))
    for i in xrange(len(reference)):
        try:
            euler1 = numpy.deg2rad(reference[i])
        except:
            logging.error("%d -> %s"%(i, str(reference.shape)))
            raise
        ang[i, :] = transforms.quaternion_from_euler(euler1[0], euler1[1], euler1[2], 'rzyz')
    n = len(reference)
    for i in xrange(align.shape[0]):
        euler1 = numpy.deg2rad(align[i, :3])
        ang[i+n, :] = transforms.quaternion_from_euler(euler1[0], euler1[1], euler1[2], 'rzyz')
    logging.info("Estimating nearest neighbors")
    neigh = manifold.knn_geodesic_cache(ang, nn, cache_file=cache_file)
    
    neighcol = neigh.col.reshape((ang.shape[0], nn+1))
    for i in xrange(len(reference)):
        avg = None
        frame = ang[neighcol[i, 0]]
        #frame = transforms.quaternion_inverse(ang[neighcol[i, 0]])
        for j in xrange(1, nn+1):
            index = neighcol[i, j]
            euler = numpy.rad2deg(transforms.euler_from_quaternion(transforms.quaternion_multiply(ang[index], frame), 'rzyz'))
            if int(index-len(reference)) < 0: continue
            img1 = ndimage_file.read_image(image_file, int(index-len(reference)))
            if 1 == 0:
                img = eman2_utility.rot_shift2D(img1, -(euler[0]+euler[2]), 0, 0, 0)
            else:
                img = scipy.ndimage.interpolation.rotate(img1, -(euler[0]+euler[2]), mode='wrap')
                diff = (img.shape[0]-img1.shape[0])
                if (diff%2) == 0:
                    diff /= 2
                    img = img[diff:(img.shape[0]-diff), diff:(img.shape[0]-diff)]
                else:
                    diff /= 2
                    img = img[diff:(img.shape[0]-diff-1), diff:(img.shape[0]-diff-1)]
            
            if avg is None: avg = img.copy()
            else: avg += img
        ndimage_file.write_image(output, avg, i)















