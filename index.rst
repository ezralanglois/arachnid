====================
Welcome to Arachnid
====================

Arachnid is an open source software package written primarily in Python to perform image
processing in single-particle reconstruction. It has the following highlighted applications
and subpackage:

- AutoPicker: Automated reference-free particle selection
- ViCer: Automated particle verification
- pySPIDER: Python batch files designed to better automate the reconstruction protocol

The algorithms and tools provided are rooted in image processing and object recognition as well as 
machine learning (see the `malibu`_ package). Critical, time-intensive sections of the code have 
been optimized in C/C++ with a SWIG interface to Python and Fortran (using f2py). This package 
is also utilizes several third-party packages, namely `SPIDER`_:, a full list 
is given in :doc:`attribution<docs/attribution>`.

This software is under development by the `Frank Lab`_ and is licensed under :doc:`GPL 2.0 <license>` 
or later.

.. raw:: html

	<a class="btn btn-lg btn-success" href="#" role="button">Download</a></p>

.. class:: btn

`Frank Lab`_

Table of Contents
==================

.. toctree::
	:maxdepth: 1

	Install <docs/install>
	License <license>
	Reconstruction Protocol <docs/reconstruction>
	Command-line Protocol <docs/reconstruction_cmdline>
	Manual <docs/manual>
	Developer's Guide <docs/development>
	API <docs/api>
	Attribution <docs/attribution>

How to cite
===========


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

This software is under development by members of the `Frank Lab`_ and supported by the Howard Hughes 
Medical Institute (HHMI) and NIH grants R37 GM 29169 and R01 GM 55440 to Prof. Joachim Frank.

We can use your support too! See the :ref:`Developer's Guide <contribute>` for ways you may 
contribute. This can be just as easy as reporting a bug or you can even add your own scripts.

References
==========

.. [Langlois2011b] `Langlois, R., Pallesen, J., and Frank, J. (2011).
                   Reference-free segmentation enhanced with data-driven template matching for particle selection in cryo-electron microscopy.
                   Journal of structural biology 175 (3) 353-361 <http://view.ncbi.nlm.nih.gov/pubmed/21708269>`_
.. [Langlois2011a] `Langlois, R., and Frank, J. (2011).
                   A clarification of the terms used in comparing semi-automated particle selection algorithms in Cryo-EM.
                   Journal of structural biology 175 (3) 348-352 <http://www.ncbi.nlm.nih.gov/pubmed/21420497>`_


.. _`Frank Lab`: http://www.columbia.edu/cu/franklab/index.html
.. _`malibu`: http://code.google.com/p/exegete
.. _`Sparx`: http://sparx-em.org/sparxwiki/
.. _`SPIDER`: http://www.wadsworth.org/spider_doc/spider/docs/



