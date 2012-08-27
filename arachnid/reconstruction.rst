=======================
Reconstruction Protocol
=======================

This protocol describes 3D reconstruction of a biological specimen (e.g. the ribosome) 
from a collection of electron micrographs.

---------
Overview
---------

.. contents:: 
	:depth: 1
	:local:
	:backlinks: none

Getting Started
===============

Now that you have finished collecting your data, you are ready to begin the image processing. At
this point you should have two things:

	#. A set of micrographs
	#. Information describing your data collection:
		- Pixel size, A
		- Electron energy, KeV
		- Spherical aberration, mm
		- Magnification
		- Actual size of particle, pixels

.. note::
	
	For Frank Lab members, you must source Arachnid to get full access to pySpider (EACH time you start a terminal window).
	
	.. sourcecode:: sh
	
		$ source /guam.raid.cluster.software/arachnid/arachnid.rc

For this tutorial, examples will be given with configuration files rather than command line
arguments. See :ref:`Running the Scripts <running-scripts>` for more information on
configuration files.

Creating a project
------------------

The `spi-project` script generates an entire pySPIDER project including
	
	- Directories
	- Configuration files

A list of options (default configuration file) can be obtain by simply running
the program without arguments.

.. sourcecode:: sh
	
	$ spi-project
	ERROR:root:Option --input-files requires a value - found empty
	#  Program:	spi-project
	#  Version:	0.0.1
	
	#  Generate all the scripts and directories for a pySPIDER project
	#  
	#  $ spi-project micrograph_files* -o project-name -r raw-reference -e extension -p params -w 4 --apix 1.2 --voltage 300 --cs 2.26 --xmag 59000 --pixel-diameter 220
	#  # or if params.ext exists
	#  $ spi-project micrograph_files* -o project-name -r raw-reference -e ext -p params -w 4
	#  

	
	input-files:                            #               (-i)    List of input filenames containing micrographs
	output:                                 #               (-o)    Output directory with project name
	raw-reference:                          #               (-r)    Raw reference volume
	ext:                                    #               (-e)    Extension for SPIDER (three characters)
	apix:                           0.0     #       Pixel size, A
	voltage:                        0.0     #       Electron energy, KeV
	cs:                             0.0     #       Spherical aberration, mm
	xmag:                           0.0     #       Magnification
	pixel-diameter:                 0       #       Actual size of particle, pixels
	...

The values shown above (for brevity only a partial list) are all required for this script to run. One
way to set these values is to create a configuration file. The name of the configuration file can
be the name of your project, e.g. `ribosome_70s`.

.. sourcecode:: sh
	
	$ spi-project > ribosome_70s.cfg
	ERROR:root:Option --input-files requires a value - found empty

You will see an error print with this method of creating a configuration file, please ignore it. You
can now edit the configuration file below. Shown in the example below are example values for each required
parameter.

.. sourcecode:: sh
	
	$ vi ribosome_70s.cfg
	#  Program:	spi-project
	#  Version:	0.0.1
	
	#  Generate all the scripts and directories for a pySPIDER project
	#  
	#  $ spi-project micrograph_files* -o project-name -r raw-reference -e extension -w 4 --apix 1.2 --voltage 300 --cs 2.26 --xmag 59000 --pixel-diameter 220
	#  

	
	input-files:               ../mic*.tif  #               (-i)    List of input filenames containing micrographs
	output:                    ribosome_70s	#               (-o)    Output directory with project name
	raw-reference:             emd_1001.map #               (-r)    Raw reference volume
	ext:                            spi     #               (-e)    Extension for SPIDER (three characters)
	is-ccd:							False	#		Set true if the micrographs were collected on a CCD (and have not been processed)
	apix:                           1.2     #       Pixel size, A
	voltage:                        300     #       Electron energy, KeV
	cs:                             2.26    #       Spherical aberration, mm
	xmag:                           59000   #       Magnification
	pixel-diameter:                 220     #       Actual size of particle, pixels
	...


	


