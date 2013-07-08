'''
.. Created on Jun 30, 2013
.. codeauthor:: robertlanglois
'''
from .. import ndimage_interpolate
import numpy, numpy.testing

full_test=False

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
