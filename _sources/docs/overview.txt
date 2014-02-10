========
Overview
========

Arachnid is an open source software package written primarily in Python that processes
images of macromolecules captured by cryo-electron microscopy (cryo-EM). Arachnid is
focused on automating the single-particle reconstruction workflow and can be thought 
of as two subpackages:
	
	1. Arachnid Prime
		A SciPy Toolkit (SciKit) that focuses on every step of the single-particle
		reconstruction workflow up to orientation assignment and classification.
	2. pySPIDER
		A library that allows one to write SPIDER batch files in Python along with
		a set of Python scripts for nearly every step in the single-particle
		reconstruction protocol.

.. note::

	The pySPIDER package requires the installation of the SPIDER software package as an 
	external dependency whereas the Arachnid Prime package is self-contained. The scripts
	that require SPIDER to function are prefixed with `sp-` while the Arachnid Prime scripts
	that have no external dependency are prefixed with `ara-`.

Arachnid has the following highlighted applications:

	- AutoPicker: Automated reference-free particle selection
	- ViCer: Automated particle verification
	
The user has several options when interfacing with Arachnid including but Graphical User
Interfaces (GUIs) and command line scripts:

	1. Overall 
		a) Wizard GUI for the preprocessing workflow
		b) Visualzation tools GUI to validate processing
	2. For individual scripts
		a) Simple Options Editor GUI
		b) Configuration files
		c) Command line options
	
The algorithms and tools provided are rooted in image processing and object recognition as well as 
machine learning (see the `malibu`_ package). Critical, time-intensive sections of the Arachnid Prime
code have been optimized in C/C++ with a SWIG interface to Python and Fortran (using f2py). This package 
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
contamination. Ryan H. Smith (Columbia University) under the additiona mentorship of Dr. Hstau Liao 
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

References
==========

.. [Langlois2014]  Langlois, R. E., Ash, J. T.,  Pallesen, J., and Frank, J. (2014).
                   Fully Automated Particle Selection and Verification in Single-Particle Cryo-EM.
                   In the proceedings of Minisymposium on Computational Methods in Three-Dimensional Microscopy Reconstruction. 
                   Birkhauser Springer (In press)
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


