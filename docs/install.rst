=============
Installation
=============

.. toctree::
	:maxdepth: 0
	:hidden:
	
	install_franklab
	Attribution <attribution>

.. note::
	
	If you are a member of the Frank lab, see :doc:`Frank Lab Install <install_franklab>`

Anaconda
========

The recommended method for most users to install Arachnid is 
to use the Anaconda package. Anaconda makes it simple to install
all the Arachnid dependencies.

Prerequisites
-------------

.. note::

	You must also install the SPIDER package. This includes a single binary 
	executable which is necessary to run the standard Arachnid workflow. It
	can be found at: http://spider.wadsworth.org/spider_doc/spider/docs/spi-register.html

The following prerequisites are generally already available on many Linux distributions
but, have been found to be absent from a few.

	- Fortran runtime libraries: libgfortran.so.3           (No longer required for the daily build)
	- Fastest Fourier transform (in the west): libfftw3.so (No longer required for the daily build)

I am currently working to address these missing dependencies. The daily builds have already
included the Fastest Fourier transform (in the west), libfortran and include SPIDER. The `ara-control`
and `ara-project` scripts should automatically find the included SPIDER executables.

Quick Start
-----------

Simply download and run this script:

	:download:`../install.sh`

.. sourcecode:: sh
	
	$ cd <to-directory you wish to install Arachnid/Anaconda>
	
	# Copy script to this same directory
	
	# Latest stable build
	
	$ sh install.sh
	
	# Latest accelerated stable build 
	# (requires Premium package free for Academic use)
	
	$ sh install.sh mkl
	
	# Latest daily build
	
	$ sh install.sh dev
	
	# Latest accelerated daily build 
	# (requires Premium package free for Academic use)
	
	$ sh install.sh dev-mkl

To update your code, do one of the following:

For the latest stable build:

.. sourcecode:: sh
	
	$ conda update arachnid
	
For the latest daily build:

.. sourcecode:: sh
	
	$ conda install arachnid-dev --force

If you want to update the accelerate versions, just add `-mkl` to the end.

For example:

.. sourcecode:: sh
	
	$ conda update arachnid-mkl


See `Speeding up the code`_ for more information about the Accelerated
builds.

Step-by-step
------------

These are the same steps run by the script in the 
`Quick Start`_ section.

#. Move to the directory you wish to install Anaconda
	
	.. sourcecode:: sh
		
		cd to-path-where-anaconda-will-be-installed/

#. Download Anaconda Miniconda installer

	.. sourcecode:: sh
	
		wget http://repo.continuum.io/miniconda/Miniconda-3.0.0-Linux-x86_64.sh

#. Install Anaconda
	
	.. sourcecode:: sh
	
		$ sh Miniconda-3.0.0-Linux-x86_64.sh -b -p $PWD/anaconda
	
	You must ensure Anaconda is on your path by doing the following
	
	.. sourcecode:: sh
	
		# Bash Shell
		$ export PATH=$PWD/anaconda/bin:$PATH 
		
		# C-Shell
		$ setenv PATH $PWD/anaconda/bin:$PATH
	
	.. note::
		
		The above step is only required the first time. Adding anaconda to your future
		terminals as follows:
		
		- For bash users, Anaconda updates your `$HOME/.bashrc`.
		
		- C/Korn-shell users (csh or tcsh) must add the above C-shell 
		  line to their own `$HOME/.cshrc`.

#. Installing Arachnid
	
	.. sourcecode:: sh
	
		conda install -c https://conda.binstar.org/public arachnid
	
	.. note::
		
		There are four versions of arachnid you can install
		#. Stable build ($ conda install arachnid)
		#. Accelerated Stable build ($ conda install arachnid-mkl)
		#. Daily build ($ conda install arachnid-dev)
		#. Accelerated Daily build ($ conda install arachnid-dev-mkl)
		
		The accelerated builds require a premium package and thus
		a license file. This license file is free for Academic use.
		See `Speeding up the code`_ for more information.
		
	
#. Simplify for future use
		
		To simplify the above command, you may add the binstar repository to
		your `$HOME/.condarc`. You may download an example file
		:download:`../.condarc` or copy the following:
		 
		.. literalinclude:: ../.condarc
   			:language: sh
   		
   		This allows you to install or update arachnid with the following
   		commands::
   			
   			conda install arachnid
   			conda update arachnid

Troubleshooting
---------------

#. Arachnid does not seem to be running or stalls!

	Run `ara-vicer`, if you see the following error message, then you are
	using the accelerated version of Arachnid and require a license file
	from continuum. Good news, this is free if you have .edu email!
	
	See `Speeding up the code`_ for more information.
	
	Alternatively, you can install a non-accelerated version of Arachnid.
	
	.. sourcecode:: sh
	
		$ ara-vicer
		
		Vendor:  Continuum Analytics, Inc.
		Package: mkl
		Message: trial mode EXPIRED 221 days ago
		
		    You cannot run mkl without a license any longer.
		    A license can be purchased it at: http://continuum.io
		    We are sorry for any inconveniences.
		
		    SHUTTING DOWN PYTHON INTERPRETER
	

#. If the installation gives the following error:

	.. sourcecode:: sh
		
		Traceback (most recent call last):
		File "/home/liaoh/anaconda/bin/conda", line 3, in <module>
		from conda.cli import main
	
	Then added `$PWD/anaconda/lib` to your `LD_LIBRARY_PATH`. This is normally not necessary.

#. If using the GUI gives the following error:

	.. sourcecode:: sh
		
		Cannot mix incompatible Qt library (version 0x40701) with this library (version 0x40704)
		/home/robertl/anaconda/bin/launcher: line 3:  9842 Aborted                 /home/robertl/anaconda/bin/python main.py $@
	
	This can be solved by updating your PySide library.

	.. sourcecode:: sh
	
		$ conda install -c https://conda.binstar.org/asmeurer pyside --yes

#. Alternative PyQt4 library for the GUI
	
	A user reported that installing the PyQt4 library using conda did not work. If you run into the
	same problem, then the workaround is to manually `download <https://binstar.org/asmeurer/pyqt/4.10.3/files>`_ the library.

#. Install the daily build to get the latest bug fixes for the current version
	
	$ conda install -c https://conda.binstar.org/public arachnid-dev --force

Speeding up the code
--------------------

#. **Optional:** Install Accelerate
	
	.. note::
	
		This step requires you obtain a license from 
		https://store.continuum.io/cshop/accelerate/. This is 
		free if you have an .edu email 
		https://store.continuum.io/cshop/academicanaconda.
	
	.. sourcecode:: sh
	
		$ conda install arachnid-mkl

Easy install
============

This is usually the fastest way to install the latest stable
release. If you have pip or easy_install, you can install or update
with the command::

    pip install -U arachnid

or::

    easy_install -U arachnid

for easy_install.


From Source
===========

Download the package from 

- Latest stable version: https://pypi.python.org/packages/source/a/arachnid/arachnid-|release|.tar.gz
- Latest development version: https://github.com/ezralanglois/arachnid/archive/master.zip

Then unpack the sources and cd into archive.

This packages uses distutils, which is the default way of installing
python modules. The install command is::

  python setup.py install



.. currentmodule:: arachnid

.. automodule:: setup


