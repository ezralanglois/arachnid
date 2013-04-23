#!/usr/bin/env python
'''
This setup file defines a build script for C/C++ or Fortran extensions.

.. Created on Aug 13, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

def configuration(parent_package='',top_path=None):  
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    import os
    
    try:
        fftw_opt = get_info('fftw',notfound_action=2)
    except: pass
    fftw_opt=dict(libraries=['fftw3f'])
    config = Configuration('spi', parent_package, top_path)
    #-ftrap=common
    if 1 == 0:
        f2py_options = ['--debug-capi']
    else: f2py_options=[]
    
    #-ffixed-form
    compiler_options=dict(define_macros=[('SP_LIBFFTW3', 1)], macros=[('SP_LIBFFTW3', 1)])#extra_f77_compiler_args=['-fdefault-real-8'],, ('SP_MP', 1)
                          #extra_f90_compiler_args=['-fdefault-real-8'])
    config.add_library('spiutil', sources=['spiutil.F90', 'spider/parabl.F90', 'spider/pksr3.F90', 'spider/ccrs.F90', 'spider/apcc.F90', 'spider/quadri.F90', 'spider/rtsq.F90', 'spider/cald.F90', 'spider/bldr.F90', 'spider/fftw3.F90', 'spider/fmrs.F90', 'spider/fmrs_2.F90', 'spider/besi1.F90', 'spider/wpro_n.F90', 'spider/prepcub.F90'], depends=['spider/CMBLOCK.INC', 'spider/FFTW3.INC'], **compiler_options) #, 'fmrs_info.mod', 'type_kinds.mod'
    fftlibs = fftw_opt['libraries']
    del fftw_opt['libraries']
    config.add_extension('_spider_reconstruct', sources=['backproject_nn4.f90', 'backproject_bp3f.f90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options)#, **fftw_opt)#, **compiler_options)
    config.add_extension('_spider_reproject', sources=['reproject.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options)
    config.add_extension('_spider_interpolate', sources=['interpolate.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options)
    #config.add_extension('_spider_rotate', sources=['rotate.pyf', 'rotate.cpp'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options)
    config.add_extension('_spider_rotate_dist', sources=['rotate.i'], define_macros=[('__STDC_FORMAT_MACROS', 1)], depends=['rotate.hpp'], swig_opts=['-c++'], libraries=['spiutil']+fftlibs)
    config.add_extension('_spider_rotate', sources=['rotate.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options)
    config.add_extension('_spider_align', sources=['align.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options)
    config.add_include_dirs(os.path.dirname(__file__))
    config.add_include_dirs(os.path.join(os.path.dirname(__file__), 'spider'))
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

