
Arachnid
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

This software is under development by the `Frank Lab`_ and is licensed under 
`GPL 2.0 <http://www.arachnid.us/license.html>`_ or later.

For more information, see `http://www.arachnid.us <http://www.arachnid.us>`_.

Alternatively, HTML documentation can be built locally using 
`python setup.py build_sphinx`, which assumes you have the prerequisite 
Python libraries. The documents can be found in `build/sphinx/html/`.

How to cite
===========

The main reference to cite is:


	Langlois, R. E., Ho D. N., Frank, J., 2014. Arachnid: Automated 
	Image-processing for Electron Microscopy. In Preparation.

See `CITE <http://www.arachnid.us/CITE.html>`_ for more information and downloadable citations.

Important links
===============

- Official source code repo: https://github.com/ezralanglois/arachnid
- HTML documentation (stable release): http://www.arachnid.us/
- Download releases: https://binstar.org/
- Issue tracker: https://github.com/ezralanglois/arachnid/issues
- Mailing list: http://groups.google.com/group/arachnid-general
- Cite: http://www.arachnid.us/CITE.html

Dependencies
============

The required dependencies to build the software are Python >= 2.6,
setuptools, Numpy >= 1.3, SciPy >= 0.7, matplotlib>=1.1.0, mpi4py>=1.2.2, 
FFTW3 or MKL, and both C/C++ and Fortran compilers.

It is also recommended you install NumPy and SciPy with an optimized Blas
library such as MKL, ACML, ATLAS or GOTOBlas.

To build the documentation, Sphinx>=1.0.4 is required.

All of these dependencies can be found in a single free binary 
package: `Anaconda`_.

Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

	python setup.py install --prefix=$HOME

Alternatively, in Anaconda you may use::

	conda pip arachnid

Development
===========

You can check out the latest source with the command::
	
	git clone https://github.com/ezralanglois/arachnid/arachnid.git

.. _`Frank Lab`: http://franklab.cpmc.columbia.edu/franklab/
.. _`Anaconda`: https://store.continuum.io/
