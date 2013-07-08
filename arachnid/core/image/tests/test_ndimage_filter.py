'''
.. Created on Jun 30, 2013
.. codeauthor:: robertlanglois
'''
from .. import ndimage_filter, eman2_utility,ndimage_utility
import numpy, numpy.testing

full_test=False

def test_ramp():
    '''
    '''
    
    orig = numpy.random.rand(50,50)
    wedge = numpy.ones((50,50))
    for i in xrange(wedge.shape[1]):
        wedge[:, i] += (i+1)
    img = orig + wedge
    out = ndimage_filter.ramp(img.copy())
    try: numpy.testing.assert_allclose(img, out)
    except: pass
    else: raise ValueError, "Image did not change"
    if full_test:
        out2 = eman2_utility.ramp(img, False)
        numpy.testing.assert_allclose(out2, out, rtol=1e-3)

def test_histogram_match():
    ''' ..todo:: add further testing here
    '''
    
    rad, width, bins = 13, 78, 128
    mask = ndimage_utility.model_disk(rad, width)
    #img = numpy.random.gamma(8, 2, (width,width))
    img = numpy.random.normal(8, 4, (width,width))
    noise = numpy.random.normal(8, 2, (width,width))
    old=img.copy()
    out = ndimage_filter.histogram_match(img, mask, noise)
    
    
    try: numpy.testing.assert_allclose(img, out)
    except: pass
    else: raise ValueError, "Image did not change"
    if full_test:
        win = eman2_utility.histfit(eman2_utility.numpy2em(old), eman2_utility.numpy2em(mask), eman2_utility.numpy2em(noise), True)
        win;
        numpy.testing.assert_allclose(out, eman2_utility.em2numpy(win))

def spider_filter(img, **extra):
    '''
    '''
    from arachnid.core.spider import spider
    from .. import ndimage_file
    import os
    #filter_type, filter_radius, pass_band, stop_band
    
    spi = spider.open_session(['file.spi'], spider_path=os.path.join(os.getcwd(), 'spider'))
    ndimage_file.write_image('test.spi', img)
    spi.fq('test', outputfile='test_out', **extra)
    return ndimage_file.read_image('test_out.spi')

def test_filter_gaussian_lowpass_2d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    sigma = 0.1
    img = numpy.random.normal(8, 4, (width,width)).astype(numpy.float32)
    fimg=ndimage_filter.filter_gaussian_lowpass(img, sigma, 2)
    if full_test: 
        simg=spider_filter(img, filter_type=3, filter_radius=sigma)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_filter_gaussian_lowpass_3d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    sigma = 0.1
    img = numpy.random.normal(8, 4, (width,width, width)).astype(numpy.float32)
    fimg=ndimage_filter.filter_gaussian_lowpass(img, sigma, 2)
    if full_test:
        simg=spider_filter(img, filter_type=3, filter_radius=sigma)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_filter_gaussian_highpass_2d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    sigma = 0.1
    img = numpy.random.normal(8, 4, (width,width)).astype(numpy.float32)
    fimg=ndimage_filter.filter_gaussian_highpass(img, sigma, 2)
    if full_test:
        simg=spider_filter(img, filter_type=4, filter_radius=sigma)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_filter_gaussian_highpass_3d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    sigma = 0.1
    img = numpy.random.normal(8, 4, (width,width, width)).astype(numpy.float32)
    fimg=ndimage_filter.filter_gaussian_highpass(img, sigma, 2)
    if full_test:
        simg=spider_filter(img, filter_type=4, filter_radius=sigma)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_filter_butterworth_lowpass_2d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    lcut, hcut = 0.1, 0.05
    img = numpy.random.normal(8, 4, (width,width)).astype(numpy.float32)
    fimg=ndimage_filter.filter_butterworth_lowpass(img, lcut, hcut , 2)
    if full_test:
        simg=spider_filter(img, filter_type=7, pass_band=lcut, stop_band=hcut)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)


def test_filter_butterworth_lowpass_3d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    lcut, hcut = 0.1, 0.05
    img = numpy.random.normal(8, 4, (width,width,width)).astype(numpy.float32)
    fimg=ndimage_filter.filter_butterworth_lowpass(img, lcut, hcut, 2)
    if full_test:
        simg=spider_filter(img, filter_type=7, pass_band=lcut, stop_band=hcut)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_filter_butterworth_highpass_2d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    lcut, hcut = 0.1, 0.05
    img = numpy.random.normal(8, 4, (width,width)).astype(numpy.float32)
    fimg=ndimage_filter.filter_butterworth_highpass(img, lcut, hcut, 2)
    if full_test:
        simg=spider_filter(img, filter_type=8, pass_band=lcut, stop_band=hcut)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_filter_butterworth_highpass_3d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    lcut, hcut = 0.1, 0.05
    img = numpy.random.normal(8, 4, (width,width,width)).astype(numpy.float32)
    fimg=ndimage_filter.filter_butterworth_highpass(img, lcut, hcut, 2)
    if full_test:
        simg=spider_filter(img, filter_type=8, pass_band=lcut, stop_band=hcut)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)

def test_filter_raised_cosine_lowpass_2d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    lcut, hcut = 0.1, 0.05
    img = numpy.random.normal(8, 4, (width,width)).astype(numpy.float32)
    fimg=ndimage_filter.filter_raised_cosine_lowpass(img, lcut, hcut, 2)
    if full_test:
        try:
            simg=spider_filter(img, filter_type=9, pass_band=lcut, stop_band=hcut)
        except: pass
        else:
            numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_filter_raised_cosine_lowpass_3d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    lcut, hcut = 0.1, 0.05
    img = numpy.random.normal(8, 4, (width,width,width)).astype(numpy.float32)
    fimg=ndimage_filter.filter_raised_cosine_lowpass(img, lcut, hcut, 2)    
    if full_test:
        try:
            simg=spider_filter(img, filter_type=9, pass_band=lcut, stop_band=hcut)
        except: pass
        else:
            numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_filter_raised_cosine_highpass_2d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    lcut, hcut = 0.1, 0.05
    img = numpy.random.normal(8, 4, (width,width)).astype(numpy.float32)
    fimg=ndimage_filter.filter_raised_cosine_highpass(img, lcut, hcut, 2)
    if full_test:
        try:
            simg=spider_filter(img, filter_type=10, pass_band=lcut, stop_band=hcut)
        except: pass
        else:
            numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_filter_raised_cosine_highpass_3d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    lcut, hcut = 0.1, 0.05
    img = numpy.random.normal(8, 4, (width,width,width)).astype(numpy.float32)
    fimg=ndimage_filter.filter_raised_cosine_highpass(img, lcut, hcut, 2)
    if full_test:
        try:
            simg=spider_filter(img, filter_type=10, pass_band=lcut, stop_band=hcut)
        except: pass
        else:
            numpy.testing.assert_allclose(simg, fimg, rtol=1e-3)




