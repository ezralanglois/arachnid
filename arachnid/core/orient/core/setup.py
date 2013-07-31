#!/usr/bin/env python
'''
This setup file defines a build script for C/C++ or Fortran extensions.

from distutils.core import setup, Extension
import numpy

setup(name='_transformations', ext_modules=[
      Extension('_transformations', ['transformations.c'],
      include_dirs=[numpy.get_include()], extra_compile_args=[])],)

.. Created on Aug 13, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

def configuration(parent_package='',top_path=None):  
    from numpy.distutils.misc_util import Configuration
    from arachnid.setup import compiler_options
    import os
    config = Configuration('core', parent_package, top_path)
    compiler_args, compiler_libraries, compiler_defs = compiler_options()[3:]
    config.add_library('_healpixlib', sources=['healpix/ang2pix_nest.c', 
                                               'healpix/ang2pix_ring.c',
                                               'healpix/pix2ang_nest.c',
                                               'healpix/pix2ang_ring.c',
                                               'healpix/nest2ring.c',
                                               'healpix/ring2nest.c',
                                               'healpix/nside2npix.c',
                                               'healpix/npix2nside.c',
                                               'healpix/mk_pix2xy.c',
                                               'healpix/mk_xy2pix.c'], depends=['healpix/chealpix.h'])
    config.add_extension('_transformations', sources=['transforms.c'])
    config.add_extension('_healpix', sources=['healpix.c'], libraries=['_healpixlib'])
    
    rot_src = 'rotation_mapping_wrap.cpp' if os.path.exists(os.path.join(os.path.dirname(__file__), 'rotation_mapping_wrap.cpp')) else 'rotation_mapping.i'
    config.add_extension('_rotation_mapping', sources=[rot_src], define_macros=[('__STDC_FORMAT_MACROS', 1)]+compiler_defs, depends=['rotation_mapping.hpp', 'linalg.hpp'], swig_opts=['-c++'], extra_link_args=compiler_args, libraries=compiler_libraries)
    config.add_include_dirs(os.path.dirname(__file__))
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

