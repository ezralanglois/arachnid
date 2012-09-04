'''
Installation Notes
==================

Arachnid depends on several packages that cannot or should not be installed
using easy_install. 

NumPY and SciPy fall into the should-not category because
they can be sped up substantially if you install a vendor tuned linear 
algebra library first.

PyQT4 and EMAN2/Sparx have a more extensive set of installation requirements
and thus must be downloaded from their respective sites.

Matplotlib can use the PyQT4 library if installed after PyQT4.

Prerequisites
=============

Please look over the list of prerequisites, if you do not have one
installed then refer to the installation steps below.

Compilers
---------

The following compilers are required if you install from the source:

    - C compiler
    - C++ compiler
    - Fortran compiler

See `NumPY Installation <http://docs.scipy.org/doc/numpy/user/install.html#building-from-source>`_ 
for details on how to compile this source.

Checking
--------

You can check what dependencies you have installed (and accessible) with the following command:

.. sourcecode:: sh

    $ python setup.py check
      running check
      Checking for mpi4py: not found
      ---
      Checking for numpy: found - 1.6.1
      Checking for scipy: found - 0.10.0rc1
      Checking for matplotlib: found - 1.1.0
      Checking for PyQt4: found
      Checking for matplotlib: found - 1.1.0
      Checking for EMAN2: found

In the above example, only `mpi4py` was not installed.

Packages to Download
--------------------

    - Vendor tuned linear algebra library (Required to run fast!)
        
        - `ACML`_
        - `MKL`_
        - `Atlas`_
        - `Lapack`_ / `Blas`_
        - `Goto Blas`_
    
    - Graphical user interface libraries (Future Requirement)
        
        - `QT4`_
        - `PyQT4`_
    
    - Single particle reconstruction package
        
        - `EMAN2/Sparx`_ (Currently Required)
        - `SPIDER`_ (Not required for installation but to run pySPIDER scripts)
    
    - Scientific Python packages (Required)
        
        - `Numpy`_
        - `Scipy`_
        - `Matplotlib`_

.. _`SPIDER`: http://www.wadsworth.org/spider_doc/spider/docs/spi-register.html
.. _`ACML`: http://developer.amd.com/cpu/Libraries/acml/Pages/default.aspx
.. _`MKL`: http://software.intel.com/en-us/intel-mkl/
.. _`Atlas`: http://math-atlas.sourceforge.net/
.. _`Lapack`: http://www.netlib.org/lapack/
.. _`Blas`: http://www.netlib.org/blas/
.. _`Goto Blas`: http://www.tacc.utexas.edu/tacc-projects/gotoblas2/
.. _`QT4`: http://qt.nokia.com/
.. _`EMAN2/Sparx`: http://ncmi.bcm.tmc.edu/ncmi/software/counter_222/software_86/
.. _`PyQT4`: http://www.riverbankcomputing.co.uk/software/pyqt/intro
.. _`Numpy`: http://sourceforge.net/projects/numpy/files/
.. _`Scipy`: http://sourceforge.net/projects/scipy/files/
.. _`Matplotlib`: http://matplotlib.sourceforge.net/
.. _`Sphinx`: http://sphinx.pocoo.org/
.. _`Py2app`: http://svn.pythonmac.org/py2app/py2app/trunk/doc/index.html
.. _`Py2exe`: http://www.py2exe.org/
.. _`Cx_Freeze`: http://cx-freeze.sourceforge.net/

Installation of Prerequisites
-----------------------------

#. Install Vendor-tuned Linear Algebra Library
        
        - `ACML`_
        - `MKL`_
        - `Atlas`_
        - `Lapack`_ / `Blas`_
        - `Goto Blas`_

    .. note ::
        
        For ACML, you need to install CBLAS from the source: http://www.netlib.org/clapack/cblas.tgz
        
        Change the line with BLLIB to the following line in the appropriate Makefile (e.g. Makefile.LINUX)
        
        BLLIB = -L/opt/acml4.4.0/gfortran64_mp/lib -lacml_mp -lacml_mv
        CFLAGS = -O3 -DADD\_ -fPIC
        
        Then invoke make:
        
        $ make
        
        And copy the resulting library to the ACML directory (if you want to follow the later steps closely)
        
        cp lib/LINUX/cblas_LINUX.a /path-to-acml/libcblas.a

#. Install Python 2.6 or 2.7

#. Install Numpy

    Create `site.cfg` in the Numpy source root and add the following values depending
    on where your vendor tuned library is install (this example is for ACML):
    
    .. sourcecode :: sh
        
        [blas]
        blas_libs = cblas, acml_mp, acml_mv
        library_dirs = /opt/acml4.4.0/gfortran64_mp/lib
        include_dirs = /opt/acml4.4.0/gfortran64_mp/include
        
        [lapack]
        lapack_libs = cblas, acml_mp, acml_mv
        library_dirs = /opt/acml4.4.0/gfortran64_mp/lib
        include_dirs = /opt/acml4.4.0/gfortran64_mp/include
        

#. Install Scipy

#. Install Matplotlib (Required for plotting functions)
    
    .. note ::

        If you plan on using the graphical user interface, install Qt4 and PyQt4 (steps 9 and 10) before installing matplotlib

#. Install EMAN2/Sparx

#. Install Qt4 (Required for graphical user interface)

#. Install PyQt4 (Required for graphical user interface)

#. Setup Environment
    
    For Bash:
    
    .. sourcecode :: sh
    
        source $PATH_TO_EMAN/EMAN2/eman2.bashrc
        
        # Setup path for BLAS Libraries
        
        # With ACML
        export BLAS_LIBS=acml:cblas:acml_mv
        export BLAS_PATH=/opt/acml4.4.0/gfortran64_mp
        export LD_LIBRARY_PATH=$BLAS_PATH/lib:$LD_LIBRARY_PATH
        
        # With PGI compiler - need to link fortran libraries otherwise you get this error: ImportError: No module named _spider_util
        export LDFLAGS="-L/opt/pgi/linux86-64/2011/REDIST -L/opt/pgi/linux86-64/2011/libso -pgf90libs $LDFLAGS"

.. Created on Sep 28, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

try: 
    import setuptools
    setuptools;
except: 
    import ez_setup
    ez_setup.use_setuptools()
    import setuptools
from numpy.distutils.core import setup
from distutils.core import Command
from distutils import log
import os, fnmatch,sys
import arachnid, arachnid.setup

# Classifiers http://pypi.python.org/pypi?%3Aaction=list_classifiers

def build_description(package, extra=None):
    '''Build a description from a Python package.
    
    This function builds a description from the __init__ of a Python package.
    
    :Parameters:

    package : str
              Name of the package
    extra : dict
            Keyword arguments to setup the package description
    
    :Returns:
    
    extra : dict
            Keyword arguments to setup the package description
    '''
    
    if extra is None: extra = {}
    description = [('name', 'project'), 'version', 'author', 'license', 'author_email', 'description', 'url', 'download_url', 'keywords', 'classifiers']#, ('long_description', 'doc')
    for d in description:
        if isinstance(d, tuple): key, field = d
        else: key, field = d, d
        if hasattr(package, "__"+field+"__"): 
            val = getattr(package, "__"+field+"__")
            if val is not None: extra[key] = val
            
    try:
        __import__(package.__name__+".setup").setup
        log.info("Root config package %s"%(package.setup.__name__))
        extra.update(package.setup.configuration(top_path='').todict())
    except: 
        log.error("No setup file found in root package to build extensions")
        raise
    extra['packages'] = setuptools.find_packages(exclude='pyspider')
    return extra

def rglob(pattern, root=os.curdir):
    '''Collect all files matching supplied filename pattern in and below supplied root directory.
    
    :Parameters:
    
    pattern : str
              Wild card pattern for file search
    root : str
           Directory root to start the search
    
    :Returns:
    
    val : list 
          List of files
    '''
    
    filenames = []
    for path, dirs, files in os.walk(os.path.abspath(root)):
        for filename in fnmatch.filter(files, pattern):
            filenames.append( os.path.join(path, filename) )
    return filenames

class check_dep(Command):
    '''
    '''
    description = "Check if dependencies are installed"

    user_options = []
    test_commands = {}
    
    def initialize_options(self): pass
    def finalize_options(self): pass

    def run(self):
        ret = 0
        packages = self.distribution.install_requires
        for v in self.distribution.extras_require.values():
            if isinstance(v, list): packages.extend(v)
            else: packages.append(v)
        # + self.distribution.setup_requires
        sep=['>', '<', '>=', '<=', '==']
        found = []
        notfound=[]
        for package in packages:
            for s in sep:
                idx = package.find(s)
                if idx > -1: 
                    package=package[:idx]
                    break
            try:    mod = __import__(package)
            except: notfound.append(package)
            else:   found.append((package, mod))
        for package in notfound:
            log.info("Checking for %s: not found"%(package))
        log.info('---')
        for package, mod in found:
            version = ' - '+mod.__version__ if hasattr(mod, '__version__') else ''
            log.info("Checking for %s: found%s"%(package, version))
        sys.exit(len(notfound))

if __name__ == '__main__':
    
    kwargs = build_description(arachnid)
    setup(entry_points = {
            'console_scripts': arachnid.setup.console_scripts,
            'gui_scripts': arachnid.setup.gui_scripts
          },
          long_description = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'README.rst')).read(),
          #test_suite = 'arachnid.testing',
          data_files=[('rst', rglob("*.rst"))],
          install_requires = [
            'numpy>=1.3.0',
            'scipy>=0.7.1',
            'matplotlib>=1.1.0',
            ],
            extras_require = {
            'MPI': 'mpi4py>=1.2.2',
            'Plotting': 'matplotlib>=1.1.0',
            'EMAN2' : ['EMAN2'],
            'PyQT' : ['PyQt4'],
            },
            setup_requires = [
            'Sphinx>=1.0.4',
            ],
            cmdclass = {'check': check_dep},
            **kwargs
    )


