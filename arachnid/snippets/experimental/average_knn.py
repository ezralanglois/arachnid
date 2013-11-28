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
from arachnid.core.metadata import format_alignment
from arachnid.core.image import ndimage_file
from arachnid.core.image import manifold
from arachnid.core.image import rotate
from arachnid.core.orient import healpix, orient_utility
import numpy, sys, logging
#import scipy.ndimage.interpolation

from arachnid.core.image import reproject
from arachnid.core.metadata import format_utility

if __name__ == "__main__":
    
    image_file = sys.argv[1]
    align_file = sys.argv[2]
    output = sys.argv[3]
    use_rtsq = False
    thread_count=16
    nn = 101
    #gdist = numpy.deg2rad(15)
    cache_file = "cache_recon"
    
    tracing.configure_logging()
    #Note: this code assuming you are average_localing a dala stack or a translated stack
    
    logging.info("Read alignment")
    files, align = format_alignment.read_alignment(align_file, image_file, use_3d=True)
    
    angs = healpix.angles(2, True)
    logging.info("Created %d,%d references"%angs.shape)
    
    logging.info("Convert Euler to quaternion")
    quat = numpy.zeros((len(align)+len(angs), 4))
    quat[:len(angs), :]=orient_utility.spider_to_quaternion(angs, True)
    quat[len(angs):, :]=orient_utility.spider_to_quaternion(align[:, :3], True)
    
    logging.info("Estimating nearest neighbors")
    neigh = manifold.knn_geodesic_cache(quat, nn, cache_file=cache_file)
    gmax, gmin = manifold.eps_range(neigh, nn)
    logging.info("Angular distance range for %d neighbors: %f - %f"%(nn, numpy.rad2deg(gmin), numpy.rad2deg(gmax)))
    
    ref_vol=None
    if 1 == 1:
        ref_vol = ndimage_file.read_image('output/vol_view_001.csv')
    
    avg = None
    neighcol = neigh.col.reshape((quat.shape[0], nn+1))
    for i in xrange(len(angs)):
        if avg is not None: avg[:]=0
        frame = angs[neighcol[i, 0], :].copy()
        for j in xrange(1, nn+1):
            index = neighcol[i, j]-len(angs)
            if i == 0: print j, align[index, :3], frame, orient_utility.euler_geodesic_distance(frame, align[index, :3])
            euler = rotate.rotate_euler(frame, align[index, :3].copy())
            
            sindex = files[1][index, 1]
            img1 = ndimage_file.read_image(image_file, int(sindex))
            img = rotate.rotate_image(img1, -(euler[0]+euler[2]))
            
            '''
            img = scipy.ndimage.interpolation.rotate(img1, -(euler[0]+euler[2]), mode='wrap')
            diff = (img.shape[0]-img1.shape[0])
            if (diff%2) == 0:
                diff /= 2
                img = img[diff:(img.shape[0]-diff), diff:(img.shape[0]-diff)]
            else:
                diff /= 2
                img = img[diff:(img.shape[0]-diff-1), diff:(img.shape[0]-diff-1)]
            '''
            if avg is None: avg = img.copy()
            else: avg += img
        
        ndimage_file.write_image(output, avg, i)
        if ref_vol is not None:
            avg = reproject.reproject_3q_single(ref_vol, img.shape[0]/2, frame.reshape((1, 3)))[0]
            ndimage_file.write_image(format_utility.add_prefix(output, 'ref'), avg, i)




