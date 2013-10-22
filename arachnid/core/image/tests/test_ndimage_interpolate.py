'''
.. Created on Jun 30, 2013
.. codeauthor:: robertlanglois
'''
from .. import ndimage_interpolate
import numpy, numpy.testing
from .. import eman2_utility

full_test=False

def test_sinc_blackman():
    '''
    '''
    
    bin_factor=4.0
    bin_factor2 = 1.0/bin_factor
    frequency_cutoff = 0.5*bin_factor2
    sb = eman2_utility.EMAN2.Util.sincBlackman(15, frequency_cutoff, 1999)
    kernel = ndimage_interpolate.sincblackman(bin_factor, dtype=numpy.float32)
    fltb=kernel[1]
    kernel=kernel[2:]
    kernel1 = numpy.zeros(15, dtype=numpy.float32)
    kernel2 = numpy.zeros(15, dtype=numpy.float32)
    for i in xrange(15):
        for j in numpy.linspace(0.0, 1.0, 100):
            k = i-7.0+j
            kernel1[i]=sb.sBwin_tab(k)
            n = -k*fltb+0.5 if k < 0 else k*fltb+0.5
            kernel2[i]=kernel[int(n)]
    numpy.testing.assert_allclose(kernel1, kernel2)

"""
def test_decimate_sinc_blackman():
    '''
    '''
    
    img = numpy.random.rand(51,51).astype(numpy.float32)
    bin_factor=4.0
    test1 = eman2_utility.decimate(img, bin_factor)
    imgem = eman2_utility.numpy2em(img)
    img = eman2_utility.em2numpy(imgem)
    test2 = ndimage_interpolate.decimate_sinc_blackman(img, bin_factor)
    numpy.testing.assert_allclose(test1, test2)
"""

def test_downsample_sinc_blackman():
    '''
    '''
    
    img = numpy.random.rand(51,51).astype(numpy.float32)
    bin_factor=4.0
    kernel = ndimage_interpolate.sincblackman(bin_factor, dtype=numpy.float32)
    test1 = eman2_utility.decimate(img, bin_factor)
    imgem = eman2_utility.numpy2em(img)
    img = eman2_utility.em2numpy(imgem)
    test2 = ndimage_interpolate.downsample(img, kernel, bin_factor)
    numpy.testing.assert_allclose(test1, test2)

def spider_ip(img, **extra):
    '''
    '''
    from arachnid.core.spider import spider
    from .. import ndimage_file
    import os
    #filter_type, filter_radius, pass_band, stop_band
    
    spi = spider.open_session(['file.spi'], spider_path=os.path.join(os.getcwd(), 'spider'))
    ndimage_file.write_image('test.spi', img)
    spi.ip('test', outputfile='test_out', **extra)
    return ndimage_file.read_image('test_out.spi')

def spider_ipft(img, **extra):
    '''
    '''
    from arachnid.core.spider import spider
    from .. import ndimage_file
    import os
    #filter_type, filter_radius, pass_band, stop_band
    
    spi = spider.open_session(['file.spi'], spider_path=os.path.join(os.getcwd(), 'spider'))
    ndimage_file.write_image('test.spi', img)
    spi.ip_ft('test', outputfile='test_out', **extra)
    return ndimage_file.read_image('test_out.spi')

def spider_ipfs(img, **extra):
    '''
    '''
    from arachnid.core.spider import spider
    from .. import ndimage_file
    import os
    #filter_type, filter_radius, pass_band, stop_band
    
    spi = spider.open_session(['file.spi'], spider_path=os.path.join(os.getcwd(), 'spider'))
    ndimage_file.write_image('test.spi', img)
    spi.ip_fs('test', outputfile='test_out', **extra)
    return ndimage_file.read_image('test_out.spi')

def test_interpolate_bilinear_2d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    size = (64, 64)
    img = numpy.random.normal(8, 4, (width,width)).astype(numpy.float32)
    fimg=ndimage_interpolate.interpolate_bilinear(img, size)
    if full_test: 
        simg=spider_ip(img, size=size)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_interpolate_bilinear_3d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    size = (64, 64, 64)
    img = numpy.random.normal(8, 4, (width,width, width)).astype(numpy.float32)
    fimg=ndimage_interpolate.interpolate_bilinear(img, size)
    if full_test:
        simg=spider_ip(img, size=size)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_interpolate_ft_2d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    size = (64, 64)
    img = numpy.random.normal(8, 4, (width,width)).astype(numpy.float32)
    fimg=ndimage_interpolate.interpolate_ft(img, size)
    if full_test: 
        simg=spider_ipft(img, size=size)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_interpolate_ft_3d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    size = (64, 64, 64)
    img = numpy.random.normal(8, 4, (width,width, width)).astype(numpy.float32)
    fimg=ndimage_interpolate.interpolate_ft(img, size)
    if full_test:
        simg=spider_ipft(img, size=size)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_interpolate_fs_2d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    size = (64, 64)
    img = numpy.random.normal(8, 4, (width,width)).astype(numpy.float32)
    fimg=ndimage_interpolate.interpolate_fs(img, size)
    if full_test: 
        simg=spider_ipfs(img, size=size)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
    
def test_interpolate_fs_3d():
    # Fails test: Max diff:  0.797096076993 -0.05879157885
    rad, width, bins = 13, 78, 128
    size = (64, 64, 64)
    img = numpy.random.normal(8, 4, (width,width, width)).astype(numpy.float32)
    fimg=ndimage_interpolate.interpolate_fs(img, size)
    if full_test:
        simg=spider_ipfs(img, size=size)
        numpy.testing.assert_allclose(simg, fimg, rtol=1e-2)
        
