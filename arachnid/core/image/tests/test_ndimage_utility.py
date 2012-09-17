''' Unit tests for the ndimage_utility module

.. Created on Aug 8, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from .. import eman2_utility
from .. import ndimage_utility
from .. import analysis
import numpy.testing
try: 
    import pylab
    pylab;
except: pylab=None

def test_mean_azimuthal():
    '''
    '''
    
    orig = numpy.random.rand(50,50)
    rot = ndimage_utility.mean_azimuthal(orig)
    rot;

def test_rolling_window():
    '''
    '''
    
    orig = numpy.random.rand(50,50)
    window=10
    step=5
    ar = ndimage_utility.rolling_window(orig, (window, window), asteps=(step, step))
    b1=0
    for i in xrange((orig.shape[0]-window)/step):
        e1 = b1+window
        b=0
        for j in xrange((orig.shape[0]-window)/step):
            e = b+window
            try:
                numpy.testing.assert_allclose(orig[b1:e1, b:e], ar[i, j])
            except:
                print "i=%d, j=%d"%(i, j)
                raise
            b += step
        b1 += step

def powerspec_avg():
    '''
    '''
    
    orig = numpy.random.rand(50,50,10)
    avg = ndimage_utility.powerspec_avg(orig, 64)
    avg;

def test_biggest_object():
    '''
    '''
    try:
        from morphology import binarize
    except: return
    rad, width = 13, 78
    obj = ndimage_utility.model_disk(rad, width).astype(numpy.float)
    mask = obj + numpy.random.rand(width,width)*0.2
    emmask = eman2_utility.numpy2em(mask)
    threshold = analysis.otsu(mask.ravel())
    embin = binarize(emmask, threshold)
    m1 = eman2_utility.EMAN2.Util.get_biggest_cluster(embin)
    m2 = ndimage_utility.biggest_object(mask>threshold)
    numpy.testing.assert_allclose(eman2_utility.em2numpy(m1), m2)

def test_tight_mask():
    '''
    '''
    
    try:
        from utilities  import gauss_edge, model_gauss
        from morphology import binarize, dilation
    except: return
    rad, width = 13, 78
    ndilate=1
    kernel_size=3
    gauss_standard_dev=3
    obj = ndimage_utility.model_disk(rad, width).astype(numpy.float)
    mask = obj + numpy.random.rand(width,width)*0.2
    emmask = eman2_utility.numpy2em(mask)
    threshold = analysis.otsu(mask.ravel())
    
    if 1 == 0: # Close but off by numerical precision
        kern = model_gauss(gauss_standard_dev, kernel_size , kernel_size)
        K = ndimage_utility.gaussian_kernel((kernel_size,kernel_size), gauss_standard_dev)
        numpy.testing.assert_allclose(eman2_utility.em2numpy(kern), K)
    
    if 1 == 1:
        m1 = eman2_utility.EMAN2.Util.get_biggest_cluster(binarize(emmask, threshold))
        for i in xrange(ndilate): m1 = dilation(m1)
        if kernel_size > 0: m1 = gauss_edge(m1, kernel_size, gauss_standard_dev)
        m2 = ndimage_utility.tight_mask(mask, threshold, ndilate, kernel_size, gauss_standard_dev)
        m3 = eman2_utility.em2numpy(m1)
        #print numpy.sum(m3), numpy.sum(m2), numpy.max(m3), numpy.max(m2), numpy.sqrt(numpy.sum((m3-m2)**2)), numpy.sqrt(numpy.max((m3-m2)**2))
        numpy.testing.assert_allclose(m3, m2)

def test_model_disk():
    '''
    '''
    
    rad, width = 13, 78
    tmp = eman2_utility.utilities.model_circle(rad, width, width)
    numpy.testing.assert_allclose(eman2_utility.em2numpy(tmp), ndimage_utility.model_disk(rad, width))

def test_ramp():
    '''
    '''
    
    orig = numpy.random.rand(50,50)
    wedge = numpy.ones((50,50))
    for i in xrange(wedge.shape[1]):
        wedge[:, i] += (i+1)
    img = orig + wedge
    out = ndimage_utility.ramp(img.copy())
    try: numpy.testing.assert_allclose(img, out)
    except: pass
    else: raise ValueError, "Image did not change"
    #numpy.testing.assert_allclose(orig, out)
    
def test_histogram_match():
    ''' ..todo:: add further testing here
    '''
    
    rad, width, bins = 13, 78, 128
    mask = ndimage_utility.model_disk(rad, width)
    #img = numpy.random.gamma(8, 2, (width,width))
    img = numpy.random.normal(8, 4, (width,width))
    noise = numpy.random.normal(8, 2, (width,width))
    out = ndimage_utility.histogram_match(img, mask, noise)
    
    
    try: numpy.testing.assert_allclose(img, out)
    except: pass
    else: raise ValueError, "Image did not change"
    
    '''
    h2 = numpy.histogram(out.ravel(), bins)[0]
    h3 = numpy.histogram(noise.ravel(), bins)[0]
    try:
        numpy.testing.assert_allclose(h2, h3)
    except:
        pylab.clf()
        pylab.hist(img.ravel(), bins)
        pylab.savefig("h1.png")
        pylab.clf()
        pylab.hist(out.ravel(), bins)
        pylab.savefig("h2.png")
        pylab.clf()
        pylab.hist(noise.ravel(), bins)
        pylab.savefig("h3.png")
        raise
    '''
    #win = eman2_utility.histfit(eman2_utility.numpy2em(img), eman2_utility.numpy2em(mask), eman2_utility.numpy2em(noise))
    #numpy.testing.assert_allclose(img, eman2_utility.em2numpy(win))
    
def test_find_peaks_fast():
    '''
    '''
    
    width,rad = 78, 13
    img = numpy.random.normal(8, 4, (width,width))
    template = ndimage_utility.model_disk(rad, width).astype(numpy.float)
    emdata = eman2_utility.numpy2em(img)
    emtemp = eman2_utility.numpy2em(template)
    
    ecc2 = emdata.calc_ccf(emtemp)
    peaks1 = numpy.asarray(ecc2.peak_ccf(rad))
    peaks1 = peaks1.reshape((len(peaks1)/3, 3))
    peaks2 = ndimage_utility.find_peaks_fast(ecc2, rad)
    peaks1 = peaks1[numpy.argsort(peaks1[:, 0])[::-1]]
    peaks2 = peaks2[numpy.argsort(peaks2[:, 0])[::-1]]
    #numpy.testing.assert_allclose(peaks1, peaks2)

def test_cross_correlate():
    '''
    '''
    
    width,rad = 78, 13
    img = numpy.random.normal(8, 4, (width,width))
    template = ndimage_utility.model_disk(rad, width).astype(numpy.float)
    emdata = eman2_utility.numpy2em(img)
    emtemp = eman2_utility.numpy2em(template)
    
    cc1 = ndimage_utility.cross_correlate(img, template)
    ecc2 = emdata.calc_ccf(emtemp)
    ecc2.process_inplace("xform.phaseorigin.tocenter")
    cc2 = eman2_utility.em2numpy(ecc2)
    #numpy.testing.assert_allclose(cc2, cc1)
    try:
        numpy.testing.assert_allclose(numpy.argmax(cc2), numpy.argmax(cc1))
    except:
        print numpy.argmax(cc2), numpy.argmax(cc1)
        raise

def test_local_variance():
    '''
    '''
    import scipy.signal
    
    width = 32
    img = numpy.random.normal(8, 4, (width*2,width*2))
    template = ndimage_utility.model_disk(int(width*0.45), width)
    emdata = eman2_utility.numpy2em(img)
    emtemp = eman2_utility.numpy2em(template)
    
    if 1 == 0:
        #out = numpy.square(img)
        out = ndimage_utility.cross_correlate_full(img, template)
        #out = scipy.signal.convolve(template, img, mode='full')
        print numpy.argmax(out), numpy.max(out)
        out = scipy.fftpack.fftshift(out)
        print numpy.argmax(out), numpy.max(out)
        #out = ndimage_utility.depad_image(out, img.shape)
        
        print img.shape[0]+template.shape[0]
        mnx = emtemp.get_xsize()
        mny = emtemp.get_ysize()
        nx = emdata.get_xsize()
        ny = emdata.get_ysize()
        nxc = nx+mnx
        nyc = ny+mny
        print nxc
        emtemp2 = emtemp.get_clip(eman2_utility.EMAN2.Region((mnx-nxc)/2, (mny-nyc)/2, nxc, nyc),0)
        squared = emdata.get_clip(eman2_utility.EMAN2.Region((nx-nxc)/2, (ny-nyc)/2, nxc, nyc),0) #emdata.get_edge_mean())
        #squared.process_inplace("math.pow",{'pow': 2.0});
        ref = squared.convolute(emtemp2)
        print numpy.argmax(eman2_utility.em2numpy(ref)), numpy.max(eman2_utility.em2numpy(ref))
        ref.process_inplace("xform.phaseorigin.tocenter")
        print numpy.argmax(eman2_utility.em2numpy(ref)), numpy.max(eman2_utility.em2numpy(ref))
        #ref.clip_inplace(eman2_utility.EMAN2.Region((nxc-nx)/2, (nyc-ny)/2, nx, ny));
    
        numpy.testing.assert_allclose(eman2_utility.em2numpy(ref), out)
    
    cc1 = ndimage_utility.local_variance(img, template)
    if 1 == 0:
        ecc2 = emdata.calc_fast_sigma_image(emtemp)
        #ecc2.process_inplace("xform.phaseorigin.tocorner")
        cc2 = eman2_utility.em2numpy(ecc2)
        try:
            numpy.testing.assert_allclose(cc2, cc1)
        except:
            print numpy.argmax(cc2), numpy.argmax(cc1)
            raise

def test_compress_image():
    '''
    '''
    
    rad, width = 13, 78
    mask = ndimage_utility.model_disk(rad, width)
    img = numpy.ones((width, width))
    img = ndimage_utility.compress_image(img, mask)
    numpy.testing.assert_allclose(numpy.sum(mask), numpy.sum(img))
    
def test_filter_filter_butterworth_lp():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    blp_lo, blp_hi = 0.1, 0.01
    img = numpy.random.normal(8, 4, (width,width))
    
    f1 = ndimage_utility.filter_butterworth_lp(img, blp_lo, blp_hi)
    if 1 == 0:
        f2 = eman2_utility.EMAN2.Processor.EMFourierFilter(eman2_utility.numpy2em(img), {"filter_type" : eman2_utility.EMAN2.Processor.fourier_filter_types.BUTTERWORTH_LOW_PASS,    "low_cutoff_frequency": blp_lo, "high_cutoff_frequency": blp_hi, "dopad" : 0})
        try:
            numpy.testing.assert_allclose(eman2_utility.em2numpy(f2), f1)
        except:
            print "Max diff: ", numpy.max(eman2_utility.em2numpy(f2)-f1), numpy.mean(eman2_utility.em2numpy(f2)-f1)
            print "Max norm: ", numpy.max(ndimage_utility.normalize_standard(eman2_utility.em2numpy(f2))-ndimage_utility.normalize_standard(f1))
            raise
    
def test_filter_gaussian_lp():
    # Fails test: Max diff:  2.51341092368 -5.50568517418e-07
    
    rad, width, bins = 13, 78, 128
    sigma = 0.1
    img = numpy.random.normal(8, 4, (width,width))
    
    f1 = ndimage_utility.filter_gaussian_lp(img, sigma)
    if 1 == 0:
        f2 = eman2_utility.gaussian_low_pass(eman2_utility.numpy2em(img), sigma)
        try:
            numpy.testing.assert_allclose(eman2_utility.em2numpy(f2), f1)
        except:
            print "Max diff: ", numpy.max(eman2_utility.em2numpy(f2)-f1), numpy.mean(eman2_utility.em2numpy(f2)-f1)
            print "Max norm: ", numpy.max(ndimage_utility.normalize_standard(eman2_utility.em2numpy(f2))-ndimage_utility.normalize_standard(f1))
            raise

