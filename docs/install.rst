=============
Installation
=============

.. note::
	
	If you are a member of the Frank lab, see :doc:`Frank Lab Install <install_franklab>`

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

Alternatively, you can add these commands to your $HOME/.bashrc or $HOME/.cshrc, respectively.

Additional Packages
~~~~~~~~~~~~~~~~~~~

.. sourcecode:: sh

	conda install basemap --yes			# Necessary for ara-cover
	conda install psutil --yes			# Necessary for all scripts
	conda install mysql-python --yes	# Necessary for Leginon import in GUI
	
Install Accelerate
~~~~~~~~~~~~~~~~~~

Note that this step requires you obtain a license from https://store.continuum.io/cshop/accelerate/. This is free.
if you have an .edu email.

.. sourcecode:: sh

	$ conda install accelerate --yes

Fix GUI
~~~~~~~

.. sourcecode:: sh

	$ conda install -c https://conda.binstar.org/asmeurer pyside --yes

Installing Arachnid
~~~~~~~~~~~~~~~~~~~

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


