import app.setup
import util.setup
import pyspider.setup
import core.gui.setup
import gui.setup

gui_scripts = []
console_scripts = []
console_scripts.extend(["ara-"+script for script in app.setup.console_scripts])
console_scripts.extend(["ara-"+script for script in util.setup.console_scripts])
console_scripts.extend(["sp-"+script for script in pyspider.setup.console_scripts])
gui_scripts.extend(["ara-"+script for script in core.gui.setup.gui_scripts])
gui_scripts.extend(["ara-"+script for script in gui.setup.gui_scripts])

_compiler_options=None

def ccompiler_options():
    '''
    '''
    
    #from numpy.distutils.ccompiler import new_compiler
    #from numpy.distutils.fcompiler.pg import PGroupFCompiler
    #from numpy.distutils.fcompiler.gnu import GnuFCompiler
    
    #ccompiler = new_compiler()
    # Todo test for PGI compiler
        
    openmp_enabled, needs_gomp = detect_openmp()
    compiler_args = ['-O3', '-funroll-loops'] #, '-mssse3' #, '-fast', '-Minfo=all', '-Mscalarsse', '-Mvect=sse']#, '-tp=nehalem-64']
    if openmp_enabled:
        compiler_args.append('-fopenmp')
    compiler_libraries = [] #['gomp'] if needs_gomp else []
    compiler_defs = [('USE_OPENMP', None)] if openmp_enabled else []
    return compiler_args, compiler_libraries, compiler_defs

def fcompiler_options():
    '''
    '''
    
    from numpy.distutils.fcompiler import new_fcompiler
    from numpy.distutils.fcompiler.pg import PGroupFCompiler
    from numpy.distutils.fcompiler.gnu import GnuFCompiler
    fcompiler = new_fcompiler()
    
    if issubclass(fcompiler.__class__, PGroupFCompiler):
        openmp_enabled, needs_gomp = detect_openmp()
        compiler_args = ['-fastsse', '-fast', '-Minfo=all', '-Mscalarsse', '-Mvect=sse']#, '-tp=nehalem-64']
        if openmp_enabled:
            compiler_args.append('-mp=nonuma')
        compiler_libraries = [] if needs_gomp else []
        compiler_defs = [('USE_OPENMP', None)] if openmp_enabled else []
    elif issubclass(fcompiler.__class__, GnuFCompiler):
        openmp_enabled, needs_gomp = detect_openmp()
        compiler_args = ['-O3', '-funroll-loops'] #, '--std=gnu99'
        if openmp_enabled:
            compiler_args.append('-fopenmp')
        compiler_libraries = [] if needs_gomp else []
        compiler_defs = [('USE_OPENMP', None)] if openmp_enabled else []
    else:
        raise ValueError, "Fortran compiler not supported: %s"%fcompiler.__class__.__name__
    return compiler_args, compiler_libraries, compiler_defs

def compiler_options():
    '''
    '''
    import setup, numpy
    from distutils.version import LooseVersion
    
    if _compiler_options is None:
        foptions = fcompiler_options()
        coptions = ccompiler_options()
        setup._compiler_options = foptions + coptions
        if LooseVersion(numpy.__version__) < LooseVersion('1.6.2'):
            import sys
            sys.argv.extend(['config_fc', '--f77flags="%s"'%" ".join(foptions[0]), '--f90flags="%s"'%" ".join(foptions[0])])
        
    return setup._compiler_options
    

def hasfunction(cc, funcname, add_opts=False, includes=[]):
    '''
    .. note::
        
        Adopted from https://github.com/SimTk/IRMSD/blob/master/python/setup.py
    '''
    import tempfile, os, shutil, sys
    
    tmpdir = tempfile.mkdtemp(prefix='arachnid-install-')
    devnull = oldstderr = None
    try:
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            f = open(fname, 'w')
            for inc in includes:
                f.write('#include %s\n'%inc)
            f.write('int main(void) {\n')
            f.write('    %s();\n' % funcname)
            f.write('}\n')
            f.close()
            # Redirect stderr to /dev/null to hide any error messages
            # from the compiler.
            # This will have to be changed if we ever have to check
            # for a function on Windows.
            devnull = open('/dev/null', 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            opts = ['-fopenmp'] if add_opts else []
            objects = cc.compile([fname], output_dir=tmpdir, extra_postargs=opts)
            cc.link_executable(objects, os.path.join(tmpdir, "a.out"), extra_postargs=opts)
        except:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if devnull is not None:
            devnull.close()
        shutil.rmtree(tmpdir)

def detect_openmp():
    '''
    .. note::
        
        Adopted from https://github.com/SimTk/IRMSD/blob/master/python/setup.py
    '''
    from distutils.ccompiler import new_compiler
    
    compiler = new_compiler()
    print "Attempting to autodetect OpenMP support...",
    hasopenmp = hasfunction(compiler, 'omp_get_num_threads', True, includes=['<omp.h>'])
    needs_gomp = hasopenmp
    if not hasopenmp:
        compiler.add_library('gomp')
        hasopenmp = hasfunction(compiler, 'omp_get_num_threads', includes=['<omp.h>'])
        needs_gomp = hasopenmp
    print
    if hasopenmp:
        print "Compiler supports OpenMP"
    else:
        print "Did not detect OpenMP support; parallel code disabled"
    return hasopenmp, needs_gomp
    

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('arachnid', parent_package, top_path)
    
    config.set_options(quiet=True)
    #config.add_subpackage('app')
    #config.add_subpackage('pyspider')
    #config.add_subpackage('util')
    config.add_subpackage('core')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())


