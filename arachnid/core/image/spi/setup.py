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
        fftw_opt = get_info('mkl',notfound_action=2)
    except: 
        try: 
            fftw_opt = get_info('fftw',notfound_action=2)
            #fftw_opt['libraries']=['fftw3f']
            fftw_opt['libraries'].extend(['fftw3f'])
            fftw_opt['library_dirs'].extend(['/usr/lib'])
        except: fftw_opt=dict(libraries=['fftw3f'])
    if 'library_dirs' not in fftw_opt: fftw_opt['library_dirs']=[]
    if 'include_dirs' not in fftw_opt: fftw_opt['include_dirs']=[]
    config = Configuration('spi', parent_package, top_path)
    #-ftrap=common
    if 1 == 0:
        f2py_options = ['--debug-capi']
    else: f2py_options=[]
    
    #-ffixed-form define_macros=[('SP_LIBFFTW3', 1)]+compiler_defs, 
    library_options=dict(macros=[('SP_LIBFFTW3', 1)]+compiler_defs, extra_f77_compile_args=compiler_args, extra_f90_compile_args=compiler_args)#extra_f77_compiler_args=['-fdefault-real-8'],, ('SP_MP', 1)
                          #extra_f90_compiler_args=['-fdefault-real-8'])
    config.add_library('spiutil', sources=['spiutil.F90', 'spider/tfd.F90', 'spider/fq_q.F90', 'spider/fq3_p.F90', 'spider/parabl.F90', 'spider/pksr3.F90', 'spider/fftw3.F90', 
                                           'spider/ccrs.F90', 'spider/apcc.F90', 'spider/quadri.F90', 'spider/rtsq.F90', 'spider/cald.F90', 'spider/bldr.F90', 
                                           'spider/fmrs.F90', 'spider/fmrs_2.F90', 'spider/besi1.F90', 'spider/wpro_n.F90', 'spider/prepcub.F90',
                                           'spider/fint.F90', 'spider/fint3.F90', 'spider/betai.F90', 'spider/gammln.F90', 'spider/betacf.F90', 'spider/histe.F90',
                                           'spider/interp_fbs3.F90', 'spider/interp_fbs.F90', 'spider/fbs2.F90', 'spider/fbs3.F90'], 
                                           depends=['spider/CMBLOCK.INC', 'spider/FFTW3.INC'], **library_options) #, 'fmrs_info.mod', 'type_kinds.mod'
    fftlibs = fftw_opt['libraries']+compiler_libraries
    del fftw_opt['libraries']
    config.add_extension('_spider_reconstruct', sources=['backproject_nn4.f90', 'backproject_bp3f.f90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args, library_dirs=fftw_opt['library_dirs'])
    config.add_extension('_spider_reproject', sources=['reproject.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args, library_dirs=fftw_opt['library_dirs'])
    config.add_extension('_spider_interpolate', sources=['interpolate.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args, library_dirs=fftw_opt['library_dirs'])
    config.add_extension('_spider_ctf', sources=['ctf.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args, library_dirs=fftw_opt['library_dirs'])

    #config.add_extension('_spider_interpolate', sources=['interpolate.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args, library_dirs=fftw_opt['library_dirs'])
    #-fdefault-real-8
    config.add_extension('_spider_rotate', sources=['rotate.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args, library_dirs=fftw_opt['library_dirs'])
    #config.add_extension('_spider_align', sources=['align.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args, library_dirs=fftw_opt['library_dirs'])
    config.add_extension('_spider_filter', sources=['filter.F90'], libraries=['spiutil']+fftlibs, f2py_options=f2py_options, define_macros=ccompiler_defs, extra_compile_args=ccompiler_args, extra_link_args=compiler_args, library_dirs=fftw_opt['library_dirs'])
    config.add_include_dirs(os.path.dirname(__file__))
    config.add_include_dirs(os.path.join(os.path.dirname(__file__), 'spider'))
    config.add_include_dirs(fftw_opt['include_dirs'])
    
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

