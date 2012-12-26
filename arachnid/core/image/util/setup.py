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
        blas_opt = get_info('blas_opt',notfound_action=2)
    except:
        blas_opt = get_info('blas')
    
    config = Configuration('util', parent_package, top_path)
    if 1 == 1:
        config.add_library('spider_util', sources=['spider_lib.f90'],  define_macros=[('SP_LIBFFTW3', 1)])
    else:
        config.add_library('spider_util', sources=['spider_lib.f90', 'spider/fmrs_2.f', 'spider/fmrs.f'],  define_macros=[('SP_LIBFFTW3', 1)], 
                           extra_f90_compile_args=['-x f95-cpp-input', '-fopenmp', '-D__OPENMP'],#-cpp -DSP_GFORTRAN -DSP_LIBFFTW3 -DSP_LINUX -O3 -funroll-loops -finline-limit=600 -DSP_MP -fopenmp
                           extra_f77_compile_args=['-x f95-cpp-input', '-fopenmp', '-D__OPENMP'])

    config.add_extension('_spider_util', sources=['spider_util.f90'], libraries=['spider_util'])
    config.add_extension('_image_utility', sources=['image_utility.i'], define_macros=[('__STDC_FORMAT_MACROS', 1)], depends=['image_utility.h'], swig_opts=['-c++'])
    config.add_extension('_manifold', sources=['manifold.i'], define_macros=[('__STDC_FORMAT_MACROS', 1)], depends=['manifold.hpp'], swig_opts=['-c++'], extra_info = blas_opt)
    config.add_include_dirs(os.path.dirname(__file__))
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

