=============
Installation
=============


.. currentmodule:: arachnid

.. automodule:: setup

Anaconda
========

Install Anaconda


	$ sh /guam.raid.cluster.software/arachnid/Anaconda-1.6.1-Linux-x86_64.sh -b -p $HOME/anaconda
	
Install Accelerate

Note that this step requires you obtain a license from https://store.continuum.io/cshop/accelerate/. This is free
if you have an .edu email.
	
	$ conda install accelerate --yes
	
Install Arachnid

	$ easy_install file:///guam.raid.cluster.software/arachnid/dist/arachnid-0.1.1-py2.7-linux-x86_64.egg

Fix GUI

	$ conda install -c https://conda.binstar.org/asmeurer pyside --yes



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



