''' Arachnid Cryo-EM Image Processing package

.. currentmodule:: arachnid

.. autosummary::
    :nosignatures:
    :template: api_module.rst
    
    app
    core
    pyspider
    util
    snippets

:mod:`arachnid.core`
====================

.. automodule:: arachnid.core

:mod:`arachnid.snippets`
========================

.. automodule:: arachnid.snippets

:mod:`arachnid.app`
===================

.. automodule:: arachnid.app

:mod:`arachnid.util`
====================

.. automodule:: arachnid.util

:mod:`arachnid.pyspider`
=========================

.. automodule:: arachnid.pyspider

'''

try:
    from arachnid._version import __version__ as v
    __version__ = v
    del v
except ImportError:
    __version__ = "UNKNOWN"

__project__ = "arachnid"
__author__ = "Robert Langlois"
__copyright__ = "Copyright (C) 2009-2014, Robert Langlois"
__license__ = "GPL"
__author_email__ = "rl2528@columbia.edu"
__description__ = "Single Particle Data Analysis Suite"
__url__ = "http://code.google.com/p/arachnid/"
__doc_url__ = "http://code.google.com/p/arachnid/docs/api_generated/%s.html"
#__download_url__  = "http://www.columbia.edu/cu/franklab/autopart.zip"
__keywords__ = "cryo-EM particle picking image-processing single-particle reconstruction machine learning"
__platforms__ = "linux"
__classifiers__ = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Environment :: MacOS X',
    #'Environment :: Win32 (MS Windows)',
    'Environment :: X11 Applications',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License (GPL)',
    'Natural Language :: English',
    'Programming Language :: Python',
    'Programming Language :: C',
    'Programming Language :: C++',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Cryo-electron Microscopy',
    'Topic :: Scientific/Engineering :: Single-particle reconstruction',
    #'Operating System :: Microsoft :: Windows',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Unix',
    'Operating System :: POSIX',
]