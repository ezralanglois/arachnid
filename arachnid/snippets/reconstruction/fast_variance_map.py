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
from arachnid.core.image import ndimage_file, reproject, ndimage_utility, ndimage_filter #, rotate
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
    align, header = format.read_alignment(align_file, ndarray=True)
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
        radius=90
        mask = ndimage_utility.model_disk(radius, (image_size,image_size)) #*-1+1
        glp_sigma = 2.245/15.0
        def scale_image(img, align, glp_sigma=0, mask=None, **extra):
            '''
            '''
            
            if glp_sigma > 0 and 1 == 0:
                img = ndimage_filter.gaussian_lowpass(img, glp_sigma)
            img=ndimage_utility.normalize_standard(img, mask, False)
            #img -= img.min()
            #ndimage_utility.normalize_min_max(img, out=img)
            return img
        
        if os.path.exists(format_utility.add_prefix(output, 'avg_')):
            vol = ndimage_file.read_image(format_utility.add_prefix(output, 'avg_'))
        else:
            _logger.info("Reconstructing standard map")
            iter_single_images = ndimage_file.iter_images(image_file)
            vol = reconstruct.reconstruct_bp3f_mp(iter_single_images, image_size, align, thread_count=thread_count, process_image=scale_image, glp_sigma=glp_sigma)
            
            ndimage_file.write_image(format_utility.add_prefix(output, 'avg_'), vol)
        
        #mask = ndimage_utility.tight_mask(vol, 0.00402, ndilate=3, gk_size=5, gk_sigma=2.5)[0]
        #vol *= mask
        #ndimage_file.write_image(format_utility.add_prefix(output, 'mask_avg_'), vol)

        """
        def square_mean_subtract2(img, align, refvol, glp_sigma=0, mask=None, **extra):
            '''
            '''
            
            if glp_sigma > 0:
                img = ndimage_filter.gaussian_lowpass(img, glp_sigma)
            #img -= img.min()
            avg = reproject.reproject_3q_single(refvol, img.shape[0]/2-3, align[:3].reshape((1, 3)))[0]
            if mask is not None:
                ndimage_utility.normalize_standard(avg, mask, out=avg)
            img -= avg
            img -= img.min()
            assert(numpy.alltrue(numpy.isfinite(img)))
            return img*img
            
            def compensated_variance(data):
    n = 0
    sum1 = 0
    for x in data:
        n = n + 1
        sum1 = sum1 + x
    mean = sum1/n
 
    sum2 = 0
    sum3 = 0
    for x in data:
        sum2 = sum2 + (x - mean)**2
        sum3 = sum3 + (x - mean)
    variance = (sum2 - sum3**2/n)/(n - 1)
    return variance
            
        """
        
        def square_mean_subtract2(img, align, refvol, glp_sigma=0, **extra):
            '''
            '''
            
            euler=align[:3].copy()
            avg = reproject.reproject_3q_single(refvol, img.shape[0]/2, euler.reshape((1, 3)))[0]
            img=ndimage_utility.normalize_standard(img, mask, True)
            ndimage_utility.normalize_standard(avg, mask, False, out=avg)
            numpy.subtract(img, avg, img)
            numpy.square(img, img)
            return img
        
        #import scipy.optimize
        
        #def lam_error(x0, img, avg): return img.ravel() - x0*avg.ravel()
        
        def square_mean_subtract(img, align, refvol, glp_sigma=0, **extra):
            '''
            '''
            
            euler=align[:3].copy()
            avg = reproject.reproject_3q_single(refvol, img.shape[0]/2, euler.reshape((1, 3)))[0]
            img=ndimage_utility.normalize_standard(img, mask, False)
            ndimage_utility.normalize_standard(avg, mask, False, out=avg)
            #x0 = numpy.asarray([1.0])
            #x0 = scipy.optimize.leastsq(lam_error, x0, args=(img,avg))[0]
            
            #avg *= (img*avg).sum()/(avg**2).sum()
            
            #print x0, (img*avg).sum()/(avg**2).sum()
            
            #avg *= numpy.inner(img.ravel(), avg.ravel())/numpy.linalg.norm(avg, 2)
            #img *= (img*avg).sum()/numpy.linalg.norm(img, 2)
            #avg *= img.std()/avg.std()
            #img -= img.mean()
            #avg -= avg.mean()
            #avg *= numpy.linalg.norm(img, 2)/numpy.linalg.norm(avg, 2)
            #print avg.max(), '==', img.max(), avg.mean(), '==', img.mean()
            #-0.195696661667 == 0.00570149073008
            numpy.subtract(img, avg, img)
            numpy.square(img, img)
            return img
        _logger.info("Reconstructing squared map")
        iter_single_images = ndimage_file.iter_images(image_file)
        avol=vol.T.copy()
        vol = reconstruct.reconstruct_bp3f_mp(iter_single_images, image_size, align, thread_count=thread_count, process_image=square_mean_subtract, total = len(align), refvol=avol, glp_sigma=glp_sigma, mask=mask)
        
        ndimage_file.write_image(output, vol) #*mask)
        _logger.info("Reconstructing squared map - finished")
        
        def mean_subtract(img, align, refvol, glp_sigma=0, **extra):
            '''
            '''
            
            euler=align[:3].copy()
            avg = reproject.reproject_3q_single(refvol, img.shape[0]/2, euler.reshape((1, 3)))[0]
            img=ndimage_utility.normalize_standard(img, mask, True)
            ndimage_utility.normalize_standard(avg, mask, False, out=avg)
            numpy.subtract(img, avg, img)
            return img
        
        _logger.info("Reconstructing compensated map")
        mvol = reconstruct.reconstruct_bp3f_mp(iter_single_images, image_size, align, thread_count=thread_count, process_image=mean_subtract, total = len(align), refvol=avol, glp_sigma=glp_sigma, mask=mask)
        
        vol-=(mvol**2/len(align))
        
        
        ndimage_file.write_image(format_utility.add_prefix(output, 'comp_'), vol) #*mask)
        _logger.info("Reconstructing compensated map - finished")
        
        
    
    
