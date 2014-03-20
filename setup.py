'''
Arachnid depends on several packages that cannot or should not be installed
using easy_install. 

NumPY and SciPy fall into the should-not category because
they can be sped up substantially if you install a vendor tuned linear 
algebra library first.

PySide has a more extensive set of installation requirements
and thus must be downloaded from their respective sites.

Matplotlib can use the PySide library if installed after PySide.

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

    $ python setup.py checkdep
      running check
      Checking for mpi4py: not found
      ---
      Checking for numpy: found - 1.6.1
      Checking for scipy: found - 0.10.0rc1
      Checking for matplotlib: found - 1.1.0
      Checking for PySide: found
      Checking for matplotlib: found - 1.1.0

In the above example, only `mpi4py` was not installed.

Packages to Download
--------------------

    - Vendor tuned linear algebra library (Required to run fast!)
        
        - `ACML`_
        - `MKL`_
        - `Atlas`_
        - `Lapack`_ / `Blas`_
        - `Goto Blas`_
    
    - Graphical user interface libraries (Required for GUI)
        
        - `QT4`_
        - `PySide`_
    
    - Single particle reconstruction package
        
        - `SPIDER`_ (Not required for installation but to run pySPIDER scripts)
    
    - Scientific Python packages (Required)
        
        - `Numpy`_
        - `Scipy`_
        - `Matplotlib`_
        - `scikit-learn`_
        - `scikit-image`_
        - `Basemap`_
        - `Python Imaging Library (PIL)`_
        - `mpi4py`_
    
    - Database Packages
        
        - `SQLAlchemy`_
        - `MySQL-Python`_
    
    - Other Packages
        
        - `psutil`_
        - `setuptools`_

.. _`mpi4py`: http://mpi4py.scipy.org/
.. _`SPIDER`: http://www.wadsworth.org/spider_doc/spider/docs/spi-register.html
.. _`ACML`: http://developer.amd.com/cpu/Libraries/acml/Pages/default.aspx
.. _`MKL`: http://software.intel.com/en-us/intel-mkl/
.. _`Atlas`: http://math-atlas.sourceforge.net/
.. _`Lapack`: http://www.netlib.org/lapack/
.. _`Blas`: http://www.netlib.org/blas/
.. _`Goto Blas`: http://www.tacc.utexas.edu/tacc-projects/gotoblas2/
.. _`QT4`: http://qt.nokia.com/
.. _`PySide`: http://qt-project.org/wiki/PySide
.. _`Numpy`: http://sourceforge.net/projects/numpy/files/
.. _`Scipy`: http://sourceforge.net/projects/scipy/files/
.. _`Matplotlib`: http://matplotlib.sourceforge.net/
.. _`Sphinx`: http://sphinx.pocoo.org/
.. _`scikit-image`: http://scikit-image.org/
.. _`scikit-learn`: http://scikit-learn.org/stable/
.. _`SQLAlchemy`: http://www.sqlalchemy.org/
.. _`MySQL-Python`: http://mysql-python.sourceforge.net/
.. _`Basemap`: http://matplotlib.org/basemap/
.. _`Python Imaging Library (PIL)`: http://www.pythonware.com/products/pil/
.. _`psutil`: https://code.google.com/p/psutil/
.. _`setuptools`: http://pythonhosted.org/setuptools/setuptools.html

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

#. Install setuptools

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
    
    .. note::

        If you plan on using the graphical user interface, install Qt4 and PySide (steps 9 and 10) before installing matplotlib

#. Install Qt4 (Required for graphical user interface)

#. Install PySide (Required for graphical user interface)

#. Setup Environment
    
    For Bash:
    
    .. sourcecode :: sh
            
        # Setup path for BLAS Libraries
        
        # With ACML
        export BLAS_LIBS=acml:cblas:acml_mv
        export BLAS_PATH=/opt/acml4.4.0/gfortran64_mp
        export LD_LIBRARY_PATH=$BLAS_PATH/lib:$LD_LIBRARY_PATH

.. Created on Sep 28, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

try: 
    import setuptools
    setuptools;
except: 
    import ez_setup #@UnresolvedImport
    ez_setup.use_setuptools()
    import setuptools
import arachnid.distutils.sdist
import arachnid.distutils.check_dep
from numpy.distutils.core import setup
from distutils import log
import os
import fnmatch
import arachnid.setup

# QT UI support: https://bitbucket.org/jbmohler/qtviews/src/ead44bd27b38/setup.py

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
    description = [('name', 'project'), 'version', 'author', 'license', 'author_email', 'description', 'url', 'download_url', 'keywords', 'classifiers', 'platforms']#, ('long_description', 'doc')
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
    for path, _, files in os.walk(os.path.abspath(root)):
        for filename in fnmatch.filter(files, pattern):
            filenames.append( os.path.join(path, filename) )
    return filenames

def get_readme():
    '''
    '''
    
    try: return open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'README.rst')).read()
    except: return None

if __name__ == '__main__':
    
    kwargs = build_description(arachnid)
    setup(entry_points = {
            'console_scripts': arachnid.setup.console_scripts,
            'gui_scripts': arachnid.setup.gui_scripts
          },
          long_description = get_readme(),
          data_files=[('rst', rglob("*.rst"))],
          install_requires = [
            'numpy',
            'scipy',
            'psutil',
            'scikit-learn',
            'scikit-image',
            'mpi4py>=1.2.2',
            'matplotlib>=1.1.0',
            'sqlalchemy>=0.8.2', 
            'mysql-python',
            'PIL>=1.1.7',
            'basemap',
            'setuptools', #distribute
            #'PySide',
            ],
            #extras_require = dict(pyside="PySide"), #[pyside]
            #setup_requires = [ - causes conda build to fail
            #'Sphinx>=1.0.4',
            #'nose>=1.0',
            #],
            cmdclass = {'checkdep': arachnid.distutils.check_dep.check_dep, 'sdist':arachnid.distutils.sdist.sdist},
            test_suite = 'nose.collector',
            **kwargs
    )


