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
from arachnid.core.image import ndimage_file, reproject, ndimage_utility, ndimage_filter, manifold
from arachnid.core.image import reconstruct
import logging, numpy, itertools #, functools
ndimage_utility;

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
    
    if 1 == 0:
        _logger.info("Reconstructing squared map")
        iter_single_images = ndimage_file.iter_images(image_file)
        iter_single_images = itertools.imap(lambda img: img*img, iter_single_images)
        vol2 = reconstruct.reconstruct_bp3f_mp(iter_single_images, image_size, align, thread_count=thread_count)
        
        _logger.info("Reconstructing standard map")
        iter_single_images = ndimage_file.iter_images(image_file)
        vol = reconstruct.reconstruct_bp3f_mp(iter_single_images, image_size, align, thread_count=thread_count)
        
        _logger.info("Estimating variance map")
        numpy.square(vol, vol)
        vol /= len(align)
        numpy.subtract(vol2, vol, vol)
        ndimage_file.write_image(output, vol)
    else:
        
        glp_sigma = 2.245/12.0
        def scale_image(img, align, glp_sigma=0, **extra):
            '''
            '''
            
            if glp_sigma > 0:
                img = ndimage_filter.gaussian_lowpass(img, glp_sigma)
            img -= img.min()
            #ndimage_utility.normalize_min_max(img, out=img)
            return img
        
        if os.path.exists(format_utility.add_prefix(output, 'avg_')):
            vol = ndimage_file.read_image(format_utility.add_prefix(output, 'avg_'))
        else:
            _logger.info("Reconstructing standard map")
            iter_single_images = ndimage_file.iter_images(image_file)
            vol = reconstruct.reconstruct_bp3f_mp(iter_single_images, image_size, align, thread_count=thread_count, process_image=scale_image, glp_sigma=glp_sigma)
            
            ndimage_file.write_image(format_utility.add_prefix(output, 'avg_'), vol)
        
        mask = ndimage_utility.tight_mask(vol, 0.00402, ndilate=3, gk_size=5, gk_sigma=2.5)[0]
        vol *= mask
        ndimage_file.write_image(format_utility.add_prefix(output, 'mask_avg_'), vol)
        
        if 1 == 0:
            tmp = vol.reshape((vol.shape[0], vol.shape[1]*vol.shape[2]))
            d, V = numpy.linalg.eigh(manifold.fastdot_t1(tmp, tmp, None, 1.0, 0.0))
            d = numpy.where(d < 0, 0, d)
            d /= d.sum()
            idx = d.argsort()[::-1]
            print d[:10]
            ndimage_file.write_image(format_utility.add_prefix(output, 'eig01_'), V[:, 0].reshape(vol.shape))
            ndimage_file.write_image(format_utility.add_prefix(output, 'eig02_'), V[:, 1].reshape(vol.shape))
        
        def square_mean_subtract(img, align, refvol, glp_sigma=0, **extra):
            '''
            '''
            
            if glp_sigma > 0:
                img = ndimage_filter.gaussian_lowpass(img, glp_sigma)
            #img -= img.min()
            avg = reproject.reproject_3q_single(refvol, img.shape[0]/2-3, align[:3].reshape((1, 3)))[0]
            img -= avg
            img -= img.min()
            assert(numpy.alltrue(numpy.isfinite(img)))
            return img*img
        
        _logger.info("Reconstructing squared map")
        iter_single_images = ndimage_file.iter_images(image_file)
        vol = reconstruct.reconstruct_bp3f_mp(iter_single_images, image_size, align, thread_count=thread_count, process_image=square_mean_subtract, refvol=vol, glp_sigma=glp_sigma)
        ndimage_file.write_image(output, vol*mask)
        
        
    
    
