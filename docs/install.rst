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

Quick Start
-----------

Simply download and run this script: :download:/install.sh.

.. sourcecode::
	
	$ cd <to-directory you wish to install Arachnid/Anaconda
	
	# Copy script to this same directory
	
	$ sh install.sh

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

Speeding up the code
--------------------

#. **Optional:** Install Accelerate
	
	.. note::
	
		This step requires you obtain a license from 
		https://store.continuum.io/cshop/accelerate/. This is 
		free if you have an .edu email.
	
	.. sourcecode:: sh
	
		$ conda install accelerate --yes

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


