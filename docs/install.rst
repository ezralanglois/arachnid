=============
Installation
=============


FrankLab
========


Install Anaconda

.. sourcecode:: sh

	$ cd <to-path-where-anaconda-will-be-installed>
	
	$ /guam.raid.cluster.software/arachnid/install
	
Update Arachnid

.. sourcecode:: sh

	$ conda install arachnid --yes -fq

.. note::

	This requires that you have a license for the premium accelerate package.
	
	https://store.continuum.io/cshop/academicanaconda
	
	If you have a license file on one computer, then just copy it to another.
	
	$ scp -r ~/.continuum 156.111.X.XXX:~/


Anaconda
========

Move to the directory you wish to install Anaconda

Install Anaconda

.. sourcecode:: sh

	$ cd <to-path-where-anaconda-will-be-installed>
	$ sh Anaconda-1.6.1-Linux-x86_64.sh -b -p $PWD/anaconda

Ensure Anaconda is on your path

.. sourcecode:: sh

	# Bash Shell
	$ export PATH=$PWD/anaconda/bin:$PATH 
	
	# C-Shell
	$ setenv PATH $PWD/anaconda/bin:$PATH

Alternatively, you can add these commands to your $HOME/.bashrc or $HOME/.cshrc.
	
Install Accelerate

Note that this step requires you obtain a license from https://store.continuum.io/cshop/accelerate/. This is free.
if you have an .edu email.

.. sourcecode:: sh

	$ conda install accelerate --yes

Fix GUI

.. sourcecode:: sh

	$ conda install -c https://conda.binstar.org/asmeurer pyside --yes

Installing Arachnid in Anaconda
===============================

Add the Arachnid channel to your $HOME/.condarc

.. sourcecode:: sh

	$ echo "channels:" > $HOME/.condarc
	$ echo "  - http://repo.continuum.io/pkgs/pro" >> $HOME/.condarc
	$ echo "  - http://repo.continuum.io/pkgs/free" >> $HOME/.condarc
	$ echo "  - http://repo.continuum.io/pkgs/gpl" >> $HOME/.condarc
	$ echo "  - http://guam/arachnid/dist" >> $HOME/.condarc

Install Arachnid

.. sourcecode:: sh

	$ conda install arachnid --yes


Installing an official release
==============================

Easy install
------------

This is usually the fastest way to install the latest stable
release. If you have pip or easy_install, you can install or update
with the command::

    pip install -U arachnid

or::

    easy_install -U arachnid

for easy_install.


From Source
-----------
Download the package from https://code.google.com/p/arachnid/downloads/list
, unpack the sources and cd into archive.

This packages uses distutils, which is the default way of installing
python modules. The install command is::

  python setup.py install



.. currentmodule:: arachnid

.. automodule:: setup


