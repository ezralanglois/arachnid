#!/usr/bin/env python
'''
This setup file defines a build script for C/C++ or Fortran extensions.

.. Created on Aug 13, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

def configuration(parent_package='',top_path=None):  
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    from arachnid.setup import compiler_options
    import os
    
    try:
        blas_opt = get_info('blas_opt',notfound_action=2)
    except:
        try:
            blas_opt = get_info('mkl',notfound_action=2)  
        except: blas_opt = get_info('blas')
    
    config = Configuration('util', parent_package, top_path)
    
    
    compiler_args, compiler_libraries, compiler_defs = compiler_options()[3:]
    
    
    img_src = 'image_utility_wrap.cpp' if os.path.exists(os.path.join(os.path.dirname(__file__), 'image_utility_wrap.cpp')) else 'image_utility.i'
    man_src = 'manifold_wrap.cpp' if os.path.exists(os.path.join(os.path.dirname(__file__), 'manifold_wrap.cpp')) else 'manifold.i'
    config.add_extension('_image_utility', sources=[img_src, 'radon.c'], define_macros=[('__STDC_FORMAT_MACROS', 1)]+compiler_defs, depends=['image_utility.h'], swig_opts=['-c++'], extra_compile_args=compiler_args, extra_link_args=compiler_args, libraries=compiler_libraries)
    config.add_extension('_manifold', sources=[man_src], define_macros=[('__STDC_FORMAT_MACROS', 1)]+compiler_defs, depends=['manifold.hpp'], swig_opts=['-c++'], extra_info = blas_opt, extra_compile_args=compiler_args, extra_link_args=compiler_args, libraries=compiler_libraries)
    config.add_include_dirs(os.path.dirname(__file__))
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

