========
Overview
========

Arachnid is an open source software package written primarily in Python that processes
images of macromolecules captured by cryo-electron microscopy (cryo-EM). Arachnid is
focused on automating the single-particle reconstruction workflow and can be thought 
of as two subpackages:
	
	#. Arachnid Prime
		A SciPy Toolkit (SciKit) that focuses on every step of the single-particle
		reconstruction workflow up to orientation assignment and classification. This
		toolkit also includes a set of application scripts and a workflow manager.
	
	#. pySPIDER
		This subpackage functions as an interface to the SPIDER package. It includes
		both a library of SPIDER commands and a set of application scripts to run
		a set of procedures for every step of single-particle reconstruction including
		orientation assignment but not classification.

Arachnid Prime currently focuses on automating the pre-processing of the image 
data captured by cryo-EM. For example, Arachnid has the following highlighted applications 
handle the particle-picking problem:

	- AutoPicker: Automated reference-free particle selection
	- ViCer: Automated unsupervised particle verification

The user has several options when interfacing with Arachnid including both Graphical User
Interfaces (GUIs) and command line scripts. The primary interface is a simple wizard GUI
that walks the user through setting up a preprocessing workflow. The end result of the
workflow is a set of particle images and a selection file for either pySPIDER orientation 
assignment or RELION classification/orientation assignment. The user may also run the 
individual scripts on the command line.

.. note::

	The pySPIDER package requires the installation of the SPIDER software package as an 
	external dependency whereas the Arachnid Prime package is self-contained. The scripts
	that require SPIDER to function are prefixed with `sp-` while the Arachnid Prime scripts
	that have no external dependency are prefixed with `ara-`.

The algorithms and tools provided are rooted in image processing and object recognition as well as 
machine learning (see the `malibu`_ package). Critical, time-intensive sections of the Arachnid Prime
code have been optimized in C/C++ using a SWIG interface to Python and using f2py to Fortran. Arachnid
is also utilizes several third-party packages, which are highlighted in :doc:`attribution<attribution>`.

This software is under development by the `Frank Lab`_ and is licensed under 
:doc:`GPL 2.0 <../license>` or later.

Background
==========

This project was started in 2009 by Robert Langlois as an internal software package written
for members of the `Frank Lab`_. It grew from a few machine learning scripts into a substantial
software package. The released version of this package contains only a subset of the available 
scripts, which have been published as well as their corresponding utilities.

Several summer students have contributed to this project. Jonathan Ginsburg (Yeshiva University) 
developed the initial code for pySPIDER, which replaced the archaic SPIDER scripting language 
with Python. Jordan T. Ash (Rutgers University) helped to develop a particle verification 
algorithm that makes it possible to automatically process datasets with high levels of 
contamination. Ryan H. Smith (Columbia University) under the additional mentorship of Dr. Hstau Liao 
developed a more robust algorithm to align frames of movies captured on cutting-edge direct detectors.

This will be a very active project and we look forward to the continual release of new scripts
as well as new features to the core library.

.. include:: ../CITE.rst

Support
=======

This software is under development by members of the `Frank Lab`_ and supported by the Howard Hughes 
Medical Institute (HHMI) and NIH grants R37 GM 29169 and R01 GM 55440 to Prof. Joachim Frank.

We can use your support too! See the :ref:`Developer's Guide <contribute>` for ways you may 
contribute. This can be just as easy as reporting a bug or you can even add your own scripts.


.. _`Frank Lab`: http://www.columbia.edu/cu/franklab/index.html
.. _`malibu`: http://code.google.com/p/exegete
.. _`Sparx`: http://sparx-em.org/sparxwiki/
.. _`SPIDER`: http://www.wadsworth.org/spider_doc/spider/docs/


