
===========
Development
===========

See :doc:`api` for code documentation.

Getting Source
==============

To get the latest version of the source, use the following `git <http://git-scm.com/documentation>`_ command:

.. sourcecode:: sh
	
	$ git clone http://github.com/ezralanglois/arachnid

Alternatively, you can download a zip file of the latest development release:

.. raw:: html

	<a class="btn btn-primary" href="https://github.com/ezralanglois/arachnid/archive/master.zip" role="button">Download</a>


Every time you update C or Cython code, you may run the following command (rather that reinstall)

.. sourcecode:: sh
	
	$ python setup.py build_ext --inplace
	
Every time you add a new script or the first time you get the source, you may run the following command (rather that reinstall)

.. sourcecode:: sh

	$ python setup.py develop --install-dir $BIN -m

.. _contribute:

Reporting Bugs
==============

As a user or developer, you can report bugs or request features on the `Issues Tracker <https://github.com/ezralanglois/arachnid/issues>`_. Please,
search for your issue before submitting it.

Contributing Code
=================

The preferred way to contribute to Arachnid is to fork the main repository on Github and submit a "Pull Request".

#. `Create a GitHub account <https://github.com/signup/free>`_
#. `Fork the Arachnid repository <https://github.com/ezralanglois/arachnid/fork>`_
#. Clone a local copy to your machine
	
	.. sourcecode:: sh
		
		$ git clone git@github.com:<YourLogin>/arachnid/arachnid.git
	
	Replace <YourLogin> with your username for GitHub.

#. Create a branch to hold your changes

	.. sourcecode:: sh
		
		$ git checkout -b my-changes

#. When you have finished making changes, then

	.. sourcecode:: sh
		
		$ git add modified_file1 
		$ git add modified_file2
		$ git commit -m "Comment describing your modifications"

#. Upload your changes to your fork of Arachnid on GitHub

	.. sourcecode:: sh
	
		$ git push -u origin my-feature

#. Make a request to incorporate your changes 
	
	Go your fork of Arachnid on Github and click the "Pull request" button. This
	sends your changes to the maintainers for review.

Your code should be error free and conform to the current code (also avoid `import *`). You can use the 
following tools to help ensure your code conforms to the
proper standards.

 - Basic documentation and error checking

	.. sourcecode:: sh
	
		$ easy_install pyflakes
		$ pyflakes path/to/module.py

 - Unit test coverage

	.. sourcecode:: sh
	
		$ easy_install nose coverage
		$ nosetests --with-coverage path/to/tests_for_package

Debugging C/C++ code
====================

Memory errors are the bain of every C/C++ developers existence. One important tool to finding these 
errors is `valgrind <http://valgrind.org/>`_.

#. Install valgrind

#. Download and modify the Python suppressions file: `valgrind-python.supp <http://svn.python.org/projects/python/trunk/Misc/valgrind-python.supp>`_
   See the directions in the `README.valgrind <http://svn.python.org/projects/python/trunk/Misc/README.valgrind>`_ for more information on modifying
   the suppressions.

#. Run valgrind over your code (choose a small example because there is a performance cot)

.. sourcecode:: sh

	$ valgrind -v --suppressions=valgrind-python.supp python my_test_script.py
	

Testing compilation of code
===========================

To test whether your code has any problems such as uninitalized variable, use the following flags in the environment for GCC type compilers.

.. sourcecode:: sh

	$ export CFLAGS="-Werror -Wno-unused-function -Wno-unknown-pragmas -Wno-format" CXXFLAGS="-Werror -Wno-format -Wno-unknown-pragmas -Wno-unused-function"
	$ export FFLAGS="-Werror -Wno-unused-function -Wtabs" F90FLAGS="-Werror -Wno-unused-function -Wtabs" F77FLAGS="-Werror -Wno-unused-function -Wtabs"

Packaging for Anaconda
======================

Build System: CentOS 5.10
-------------------------

The Anaconda people use CentOS 5 to build Anaconda and to ensure compatibility I recommend doing the same.

.. note::
	
	Redhat backported OpenMP to gcc 4.1. This provides both backward compatibility for gcc4 compilers
	and OpenMP threading.

Here are the steps I suggest to perform a minimal install of CentOS 5 on some platform, e.g. virutual machine.

#. Get the ISO. 

You can download CentOS 5.10 from http://isoredirect.centos.org/centos/5/isos/x86_64/

I am going through a bare minimum build so I only use the first CD: CentOS-5.10-x86_64-bin-1of9.iso

#. Install

To do a bare minimum install, I recommend unselecting all the packages. This means selecting the `Customize Now`
option when you see it and unselecting the Gnome Desktop.

.. warning::

	Do not use automatic installation in VMWare if you want a bare minimum install
	of CentOS 5.

#. Install Compilers

	.. sourcecode:: sh
		
		$ yum install gcc gcc-gfortran gcc-c++ -yq
	
#. Install Git Prerequisites

	#. Install Git

	.. sourcecode:: sh
		
		$ yum install wget -yq
		$ yum install unzip -yq
		$ yum install make -yq
		$ yum install zlib zlib-devel -yq
		$ yum install openssh-server openssh-clients openssl-devel -yq
		$ yum install curl curl-devel -yq
		$ yum install expat expat-devel
		$ yum install gettext
		$ yum install bzip2

.. One long command:
.. yum install gcc gcc-gfortran gcc-c++ unzip make zlib zlib-devel openssh-server openssh-clients openssl-devel curl curl-devel expat expat-devel gettext wget bzip2 -yq
	
#. Install Git

	.. sourcecode:: sh
		
		$ wget https://github.com/git/git/archive/master.zip
		$ mv master master.zip
		$ unzip master.zip 
		$ cd git-master/
		$ make prefix=/usr install

.. One long command:
.. wget https://github.com/git/git/archive/master.zip;mv master master.zip;unzip master.zip;cd git-master/;make prefix=/usr install

#. Install Anaconda
	
	.. sourcecode:: sh
		
		$ wget http://repo.continuum.io/miniconda/Miniconda-3.0.0-Linux-x86_64.sh
		$ sh Miniconda-3.0.0-Linux-x86_64.sh -b -p $PWD/anaconda

#. Get Arachnid Source

	.. sourcecode:: sh
		
		$ git clone http://github.com/ezralanglois/arachnid/

Prerequisites
-------------

#. Install required packages

.. sourcecode:: sh

	# Install the conda build command and its dependencies
	
	conda install conda-build jinja2 setuptools binstar patchelf --yes

#. Setup Arachnid dependencies

You must add `https:/conda.binstar.org/ezralanglois` to your `$HOME/.condarc`.

See the example below:

.. literalinclude:: ../.condarc
   	:language: sh

You may download an example file :download:`../.condarc` or copy the previous.

Stable Release
--------------

.. note::

	To run these commands, https://conda.binstar.org/ezralanglois must
	be in your $HOME/.condarc.
	
	If you do not have a condarc, then a quick fix is to run the
	following:
	
	.. sourcecode:: sh
		
		$ cd <arachnid-source-directory>
		$ cp .condarc ~/

.. sourcecode:: sh
	
	$ cd <arachnid-source-directory>
	
	# Build a binary release for Anaconda
	
	$ conda build conda-recipes/stable
	
	# Upload to binstar.org repository
	
	$ binstar upload $HOME/anaconda/conda-bld/linux-64/arachnid-0.1.4-py27_1.tar.bz2
	
.. note::
	
	The accelerate (aka MKL) releases can be done the same way
	except its `conda-recipes/stable-mkl`.

Development Release (Daily Build)
---------------------------------

.. sourcecode:: sh
	
	$ cd <arachnid-source-directory>
	
	# Build a binary release for Anaconda
	
	$ conda build conda-recipes/daily
	
	# Upload to binstar.org repository (The `--force` is required since the version does not necessary change)
	
	$ binstar upload /home/robertl/anaconda/conda-bld/linux-64/arachnid-dev-0.1.5.dev-py27_1.tar.bz2 --force

.. note::
	
	The accelerate (aka MKL) releases can be done the same way
	except its `conda-recipes/daily-mkl`.


Documentation Hack
==================

To get the documentation to build correctly, you need to edit `sphinx/ext/autosummary/generate.py` in your site-packages
directory.

Change Line 143 from

.. sourcecode:: py

	for name in dir(obj):

to

.. sourcecode:: py

	for name in vars(obj):

A little background: The default autosummary code gets all inherited members of a class. This ensures only the current
members will be documented.

.. snippet tutorial

Create a Program Script
=======================

Arachnid supports two types of scripting: applications and fast prototyping. Applications are scripts
that are designed for a user and to fit on the reconstruction workflow. 

Basic Program
-------------

.. include:: /arachnid/core/app/program.py
	:start-after: beg-dev
	:end-before: end-dev

File Processor
--------------

.. include:: /arachnid/core/app/file_processor.py
	:start-after: beg-dev
	:end-before: end-dev
	
Options
-------

.. include:: /arachnid/core/app/settings.py
	:start-after: beg-dev
	:end-before: end-dev

Logging
-------

.. include:: /arachnid/core/app/tracing.py
	:start-after: beg-dev
	:end-before: end-dev

.. _add-to-workflow:

Adding a script to the workflow
-------------------------------

A workflow module should contain the following function:

.. py:function:: supports(files, **extra)

   Test if this module is required in the project workflow

   :param files: List of filenames to test
   :param extra: Unused keyword arguments
   :returns: True if this module should be added to the workflow

Basic Snippets
==============

Fast prototyping is a way to test new ideas, perform a simple task quickly or 
customize objects such as plots.

The :py:mod:`snippets` cover a set of examples for writing fast-prototype code. These
examples range from unstacking a SPIDER image stack to customizing the FSC plot.

Below are some very short snippets illustrating basic utilities in Arachnid.

Reading/Writing images
----------------------

.. include:: /arachnid/core/image/ndimage_file.py
	:start-after: beg-dev
	:end-before: end-dev

Reading/Writing metadata
------------------------

.. include:: /arachnid/core/metadata/format.py
	:start-after: beg-dev
	:end-before: end-dev


.. multi-processing
.. mpi
.. metadata
.. image
.. pyspider

.. example batch program
.. example file processor program

.. Under construction


.. ------
.. TODO
.. ------

.. todolist



