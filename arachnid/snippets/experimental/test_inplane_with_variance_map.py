''' Reconstruct a fast variance map from a subset of projections

Download to edit and run: :download:`fast_variance_map.py <../../arachnid/snippets/fast_variance_map.py>`

To run:

.. sourcecode:: sh
    
    $ python fast_variance_map.py

.. literalinclude:: ../../arachnid/snippets/fast_variance_map.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys, os
#sys.path.append('/home/robertl/tmp/arachnid-0.0.1/')

from arachnid.core.app import tracing
from arachnid.core.metadata import format, format_utility
from arachnid.core.image import ndimage_file, reproject, ndimage_utility, rotate
from arachnid.core.orient import healpix #,orient_utility
from arachnid.core.image import reconstruct
import logging, numpy #, functools
ndimage_utility;
rotate;

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    
    image_file = sys.argv[1]   # phase_flip_dala_stack.spi
    align_file = sys.argv[2]   # align.spi
    output = sys.argv[3]       # raw_vol.spi
    thread_count = int(sys.argv[4]) # Number of threads
    
    tracing.configure_logging(log_level=4)
    _logger.info("Reading alignment file")
    align = format.read_alignment(align_file)
    align,header = format_utility.tuple2numpy(align)
    image_size = ndimage_file.read_image(image_file).shape[0]
    

    radius=90
    mask = ndimage_utility.model_disk(radius, (image_size,image_size)) #*-1+1
    glp_sigma = 2.245/15.0
    def scale_image(img, align, mask=None, **extra):
        '''
        '''
        
        img=ndimage_utility.normalize_standard(img, mask, False)
        return img
    
    if os.path.exists(format_utility.add_prefix(output, 'avg_')):
        vol = ndimage_file.read_image(format_utility.add_prefix(output, 'avg_'))
    else:
        _logger.info("Reconstructing standard map")
        iter_single_images = ndimage_file.iter_images(image_file)
        vol = reconstruct.reconstruct_bp3f_mp(iter_single_images, image_size, align, thread_count=thread_count, process_image=scale_image)
        
        ndimage_file.write_image(format_utility.add_prefix(output, 'avg_'), vol)
    
    def square_mean_subtract(img, align, refvol, mask, angs, **extra):
        '''
        '''
        
        frame=align[:3].copy()
        sum = img.copy()
        sum[:]=0
        img=ndimage_utility.normalize_standard(img, mask, True)
        tmp=None
        #img = rotate.rotate_image(img, -frame[0]) # new
        frame[0]=0
        #frame2 = frame.copy()# new
        frame[:] = -frame[::-1]
        for i, ang in enumerate(angs):
            ang2 = ang.copy()
            ang2[0]=0
            euler = rotate.rotate_euler(frame, ang2)
            avg = reproject.reproject_3q_single(refvol, img.shape[0]/2, euler.reshape((1, 3)))[0]
            #euler = rotate.rotate_euler(frame2, euler) # new
            euler = rotate.rotate_euler(align[:3].copy(), euler)
            avg = rotate.rotate_image(avg, -(euler[0]+euler[2]))
            ndimage_utility.normalize_standard(avg, mask, False, out=avg)
            tmp=numpy.subtract(img, avg, tmp)
            numpy.square(tmp, tmp)
            sum+=tmp
        return sum/len(angs)
    
    ang = healpix.angles(5, True, out=numpy.zeros((10,3)))
    _logger.info("Reconstructing squared map")
    iter_single_images = ndimage_file.iter_images(image_file)
    avol=vol.T.copy()
    vol = reconstruct.reconstruct_bp3f_mp(iter_single_images, image_size, align, thread_count=thread_count, process_image=square_mean_subtract, refvol=avol, mask=mask, angs=ang)
    
    ndimage_file.write_image(output, vol) #*mask)
    _logger.info("Reconstructing squared map - finished")
        
       
        
        
    