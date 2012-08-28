
====================
Welcome to Arachnid
====================

Arachnid is an open source software package written primarily in Python to assist in 
particle selection, an essential step in single particle reconstruction. The algorithms
and tools provided are rooted in image processing and object recognition as well as machine
learning (see the `malibu`_ package). Critical, time-intensive sections of the code have 
been optimized in C/C++ with a SWIG interface to Python and Fortran (using f2py). This package 
is also utilizes several third-party packages, namely EMAN2 and SPIDER: a full list is given 
in attribution. 

This software is licensed under :doc:`GPL 2.0 <license>` or later.

Table of Contents
==================

.. toctree::
	:maxdepth: 1

	Install <install>
	License <license>
	Reconstruction Protocol <arachnid/reconstruction>
	Manual <arachnid/manual>
	Developer's Guide <arachnid/index>
	Attribution <attribution>

Todo for Launch
===============

0. Test all code and documentation
1. Push source to code.google.com
2. Package source zip for code.google.com `downloads`
3. Publish source to PyPI
4. Publish documentation to website (columbia local?)
5. Announce on 3dem-request@ncmir.ucsd.edu
6. Create code.google.com/arachnid-docs with SVN - http://manjeetdahiya.com/2010/09/29/serving-html-documentation-from-google-code-svn/
7. Test MPI
8. Test autopick dog removal bug
9. Create script to publish code, source zip, documentation and ?binary?
10. Script building tutorial (general)

Important links
===============

- Official source code repo: https://code.google.com/p/arachnid/
- HTML documentation (stable release): https://code.google.com/p/arachnid/w/list
- Download releases: https://code.google.com/p/arachnid/downloads/list
- Issue tracker: https://code.google.com/p/arachnid/issues/list
- Mailing list: http://groups.google.com/group/arachnid-general

Background
==========

This project was started in 2009 by Robert Langlois as an internal software package written
for members of the `Frank Lab`_. It grew from a few machine learning scripts into a substantial
package. The released version of this package contains only a subset of the available scripts,
which have been published as well as their corresponding utilities.

This will be a very active project and we look forward to the continual release of new scripts
as well as new features to the core library.

Support
=======

This software was developed by the `Frank Lab`_ and supported by the Howard Hughes Medical Institute
and NIH grants R37 GM 29169 and R01 GM 55440 to Prof. Joachim Frank.

We can use your support too! See the :ref:`Developer's Guide <contribute>` for ways you may 
contribute. This can be just as easy as reporting a bug or you can even add your own scripts.

References
==========

.. [Langlois2010] Langlois, R., Pallesen, J., and Frank, J. (2010).
                  Reference-free segmentation enhanced with data-driven template matching for particle selection in cryo-electron microscopy.


.. _`Frank Lab`: http://www.columbia.edu/cu/franklab/index.html
.. _`malibu`: http://code.google.com/p/exegete

Future
======

 - Pytables - Data access
 - Pypng
 - Pyjpg
 - PIL
 - Read/Write mrc/spider
 - Subclass ndarray

