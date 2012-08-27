
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

This software was developed by the `Frank Lab`_ and supported by the Howard Hughes Medical Institute
and NIH grants R37 GM 29169 and R01 GM 55440 to J. Frank.

This software is licensed under :doc:`GPL 2.0 <license>` or later.

References
==========

.. [Langlois2010] Langlois, R., Pallesen, J., and Frank, J. (2010).
                  Reference-free segmentation enhanced with data-driven template matching for particle selection in cryo-electron microscopy.


.. _`Frank Lab`: http://www.columbia.edu/cu/franklab/index.html
.. _`malibu`: http://code.google.com/p/exegete


Todo
====

0. Test all code and documentation
1. Push source to code.google.com
2. Package source zip for code.google.com `downloads`
3. Publish source to PyPI
4. Publish documentation to website (columbia local?)
5. Announce on 3dem-request@ncmir.ucsd.edu
6. Create code.google.com/arachnid-docs with SVN - http://manjeetdahiya.com/2010/09/29/serving-html-documentation-from-google-code-svn/
7. Test MPI
8. Test autopick dog removal bug
9. Test lfcpick, limited 200 windows
10. Test multi-processing bugs
11. Create script to publish code, source zip, documentation and ?binary?
12. Reconstruction protocol
13. Script building tutorial (general)

AutoPicker
----------
14. Test pca, now 0.9
15. Multi-processing bug - missing one micrograph - why?

.. todolist::

Future
======

 - Pytables - Data access
 - Pypng
 - Pyjpg
 - PIL
 - Read/Write mrc/spider
 - Subclass ndarray

