#!/usr/bin/env python
'''
This setup file defines a build script for C/C++ or Fortran extensions.

.. Created on Aug 13, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

def configuration(parent_package='',top_path=None):  
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    from arachnid.setup import compiler_options #detect_openmp
    import os
    
    compiler_args, compiler_libraries, compiler_defs, ccompiler_args, ccompiler_lib, ccompiler_defs = compiler_options()[:6]
    ccompiler_lib;
    
    
    try:
        fftw_opt = get_info('fftw',notfound_action=2)
    except: pass
    fftw_opt=dict(libraries=['fftw3f'])
    config = Configuration('spi', parent_package, top_path)
    #-ftrap=common
    if 1 == 0:
        f2py_options = ['--debug-capi']
    else: f2py_options=[]
    
    #-ffixed-form define_macros=[('SP_LIBFFTW3', 1)]+compiler_defs, 
    library_options=dict(macros=[('SP_LIBFFTW3', 1)]+compiler_defs, extra_f77_compile_args=compiler_args, extra_f90_compile_args=compiler_args)#extra_f77_compiler_args=['-fdefault-real-8'],, ('SP_MP', 1)
                          #extra_f90_compiler_args=['-fdefault-real-8'])
    config.add_library('spiutil', sources=['spiutil.F90', 'spider/fq_q.F', 'spider/fq3_p.F', 'spider/parabl.F', 'spider/pksr3.F', 'spider/fftw3.F', 
                                           'spider/ccrs.F', 'spider/apcc.F', 'spider/quadri.F', 'spider/rtsq.F', 'spider/cald.F', 'spider/bldr.F', 
                                           'spider/fmrs.F', 'spider/fmrs_2.F', 'spider/besi1.F', 'spider/wpro_n.F', 'spider/prepcub.F',
                                           'spider/fint.F', 'spider/fint3.F', 'spider/betai.F', 'spider/gammln.F', 'spider/betacf.F', 'spider/histe.F'], 
                                           depends=['spider/CMBLOCK.INC', 'spider/FFTW3.INC'], **library_options) #, 'fmrs_info.mod', 'type_kinds.mod'
    fftlibs = fftw_opt['libraries']+compiler_libraries
    del fftw_opt['libraries']
    
    #config.add_library('spi_reconstruct', sources=['backproject_nn4.f90', 'backproject_bp3f.f90'], **library_options)
    #config.add_extension('_spider_reconstruct', sources=['backproject.pyf'], libraries=['spiutil', 'spi_reconstruct']+fftlibs)#, extra_compile_args=ccompiler_args, extra_link_args=ccompiler_args)
    
    config.add_extension('_spider_reconstruct', sources=['backproject_nn4.f90', 'backproject_bp3f.f90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args)
    config.add_extension('_spider_reproject', sources=['reproject.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args)
    config.add_extension('_spider_interpolate', sources=['interpolate.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args)
    config.add_extension('_spider_rotate_dist', sources=['rotate.i'], define_macros=[('__STDC_FORMAT_MACROS', 1)]+ccompiler_defs, depends=['rotate.hpp'], swig_opts=['-c++'], libraries=['spiutil']+fftlibs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args)
    config.add_extension('_spider_rotate', sources=['rotate.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args)
    config.add_extension('_spider_align', sources=['align.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args)
    config.add_extension('_spider_filter', sources=['filter.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args)
    config.add_include_dirs(os.path.dirname(__file__))
    config.add_include_dirs(os.path.join(os.path.dirname(__file__), 'spider'))
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

