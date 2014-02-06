#!/usr/bin/env python
'''
This setup file defines a build script for C/C++ or Fortran extensions.

.. Created on Aug 13, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

def configuration(parent_package='',top_path=None):  
    from numpy.distutils.misc_util import Configuration
    from arachnid.setup import compiler_options
    from numpy.distutils.system_info import get_info
    
    try:blas_opt = get_info('mkl',notfound_action=2)  
    except: blas_opt = {}
    
    config = Configuration('core', parent_package, top_path)
    compiler_args, compiler_libraries, compiler_defs = compiler_options()[3:]
    config.add_extension('_omp', sources=['omp.c'], extra_link_args=compiler_args, define_macros=compiler_defs, extra_compile_args=compiler_args, extra_info = blas_opt)#, libraries=compiler_libraries
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

