=============
Installation
=============


FrankLab
========

If you are a member of FrankLab, then you only require to do the following in this
section.

Install Anaconda

.. sourcecode:: sh

	$ cd <to-path-where-anaconda-will-be-installed>
	# For example
	$ cd /data/robertl/
	
	$ bash /guam.raid.cluster.software/arachnid/install
	
Update to the Latest Version of Arachnid

.. sourcecode:: sh

	$ conda install arachnid --yes -fq

Update Arachnid to a specific version

.. sourcecode:: sh

	$ conda install arachnid=0.1.2 --yes -fq
	# or 
	$ conda install arachnid=0.1.3 --yes -fq

.. note::

	This requires that you have a license for the premium accelerate package.
	
	https://store.continuum.io/cshop/academicanaconda
	
	If you have a license file on one computer, then just copy it to another.
	
	$ scp -r ~/.continuum 156.111.X.XXX:~/

