.. -*- mode: rst -*-

arachnid
========

Arachnid is Python software package for image processing in single particle reconstruction of
images collected by cryo-electron microscopy. It is built on top of SciPy, `EMAN2`_/`Sparx`_ and
is distributed under the `GPL 2.0`_ license.

This project was started in 2009 by Robert Langlois as an internal software package written
for members of the `Frank Lab`_. This package contains only the published algorithms and
corresponding utilities.


For more information, see the documents in `build/sphinx/html/`. HTML documentation can be built using
`python setup.py build_sphinx`, which assumes you have the prerequisite Python libraries.


Important links
===============

- Official source code repo: https://code.google.com/p/arachnid/
- HTML documentation (stable release): https://code.google.com/p/arachnid/w/list
- Download releases: https://code.google.com/p/arachnid/downloads/list
- Issue tracker: https://code.google.com/p/arachnid/issues/list
- Mailing list: http://groups.google.com/group/arachnid-general

Dependencies
============

The required dependencies to build the software are Python >= 2.6,
setuptools, Numpy >= 1.3, SciPy >= 0.7, matplotlib>=1.1.0, mpi4py>=1.2.2,
and a working C/C++ compiler.

The required dependencies to run code in this project also includes 
EMAN2/Sparx.

To build the documentation, Sphinx>=1.0.4 is required.

Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

	python setup.py install

Development
===========

You can check out the latest source with the command::
	
	git clone https://code.google.com/p/arachnid/

.. _`Frank Lab`: http://franklab.cpmc.columbia.edu/franklab/
.. _`Sparx`: http://sparx-em.org/sparxwiki/Installer
.. _`EMAN2`: http://blake.bcm.edu/emanwiki/
.. _`GPL 2.0`: http://www.gnu.org/licenses/gpl-2.0.html
