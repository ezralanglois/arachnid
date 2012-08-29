=======================
Reconstruction Protocol
=======================

This protocol describes 3D reconstruction of a biological specimen (e.g. the ribosome) 
from a collection of electron micrographs.

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
		- Actual size of particle, pixels

.. note::
	
	For Frank Lab members, you must source Arachnid to get full access to pySpider (EACH time you start a terminal window).
	
	.. sourcecode:: sh
	
		$ source /guam.raid.cluster.software/arachnid/arachnid.rc

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
	#  $ spi-project micrograph_files* -o project-name -r raw-reference -e extension -w 4 --apix 1.2 --voltage 300 --cs 2.26 --pixel-diameter 220 --scatter-doc ribosome
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

The values shown above (for brevity only a partial list covering all required parameters) are all 
required for this script to run.

The values for each option can be set as follows:

.. sourcecode:: sh
	
	$ spi-project ../mic*.tif -o ribosome_70s -r emd_1001.map -e spi -w 4 --apix 1.2 --voltage 300 --cs 2.26 --pixel-diameter 220 --scatter-doc ribosome

Let's look at each parameter on the command line above.

The `../mic*.tif` is a list of micrographs. The shell in most operating systems understands that `*` is a wildcard 
character that allows you to select all files in directory `../` that start with `mic` and end with `.tif`. You do
not need to convert the micrographs to SPIDER format, that will be taken care of for you. In fact, the micrographs
are not converted at all, only the output particle projection windows are required to be in SPIDER format for
pySPIDER.

The `-o ribosome_70s` defines the name of the root output directory, which in this case is `ribosome_70s`. A set of
directories and configuration files/scripts will be created in this output directory (:ref:`see below <project-directory>`).

The `-r emd_1001.map` defines the raw reference volume. Ideally, this will be in MRC format with the pixel-size in the header. If not,
then you will need to edit the :py:mod:`reference` script to set the pixel size.

The `-e spi` defines the extension used in the SPIDER project. This is required by SPIDER and should be three characters.

The `-w 4` defines the number of cores to use for parallel processing.

The `-apix 1.2`, `--voltage 300`, `--cs 2.26`, and `--pixel-diameter 220` microscope parameters that define the experiment.

The `--scatter-doc ribosome` will download a ribosome scattering file to 8A, otherwise you should specify an existing scattering file
or nothing.

.. _project-directory:

The command above will create a directory called `ribosome_70s` with the following structure:

.. sourcecode:: sh

	$ ls -R ribosome_70s
	ribosome_70s/:
	cluster  local run_cluster run_local
	
	ribosome_70s/cluster:
	align.cfg  data  refine.cfg  refinement  win
	
	ribosome_70s/cluster/data:
	paramslm1
	
	ribosome_70s/cluster/refinement:
	
	ribosome_70s/cluster/win:
	
	ribosome_70s/local:
	autopick.cfg  coords  crop.cfg  defocus.cfg  pow  reference.cfg
	
	ribosome_70s/local/coords:
	
	ribosome_70s/local/pow:

In the `ribosome_70s` directory, you will find two scripts: one to invoke all local scripts and one
to invoke the cluster scripts.

Running Local Scripts
---------------------

To run all the local scripts in the proper order, use the following suggested command:

.. sourcecode:: sh

	$ cd ribosome_70s
	
	$ nohup sh run_local  > /dev/null &

.. note::
	
	All paths are setup relative to you executing a script from the project directory, e.g. `ribosome_70s`.

Running Cluster Scripts
-----------------------

Running scripts on the cluster is slightly more complicated. The `spi-project` script tries to guess the proper command
under the following assumptions:

 #. Your account is setup to run an MPI job on the cluster
 #. You have a machinefile for MPI
 #. You have SSH-AGENT or some non-password enabled setup
 #. Your cluster does not use a schdueling system like PBS or Torque

If your files are not accessible to the cluster, then you only need to copy the `cluster` directory and the
`run_cluster` script to the cluster. 

.. sourcecode:: sh

	$ cd ribosome_70s
	
	$ scp -r cluster run_cluster username@cluster:~/ribosome_70s

To run all cluster scripts in the proper order, use the following suggested command:

.. sourcecode:: sh

	$ cd ribosome_70s
	
	$ nohup sh run_cluster > /dev/null &

.. note::

	You will find your refined, amplitude-enhanced volume in `ribosome_70s/cluster/refinement` with the 
	name (assuming you specified `scattering-doc` with the appropriate file): e.g. after 13 iterations 
	of refinement, it will be called `enh_align_0013.spi`.

Improving the Data Treatment
============================

Under construction

Micrograph screening
--------------------

This can be done with `SPIDER's Montage Viewer <http://www.wadsworth.org/spider_doc/spire/doc/guitools/montage.html>`_.

Power spectra screening
-----------------------

This can be done with `SPIDER's Montage Viewer <http://www.wadsworth.org/spider_doc/spire/doc/guitools/montage.html>`_.
	
Manual CTF fitting
------------------

This can be done with `SPIDER's CTFMatch <http://www.wadsworth.org/spider_doc/spire/doc/guitools/ctfmatch/ctfmatch.html>`_. CTFMatch
will write out a new defocus file; it is recommended that you rename the current defocus file first, then save
the new defocus file with the original name of the current defocus file.

View average screening
----------------------

Generate View averages
++++++++++++++++++++++

- Reference-base: Alignment

- Reference-free: Relion

	See :mod:`arachnid.util.relion_selection` for a script to help you setup
	a selection file for Relion.

View-editing
++++++++++++

This can be done with `SPIDER's Montage Viewer <http://www.wadsworth.org/spider_doc/spire/doc/guitools/montage.html>`_.


Chimera Tricks
==============

Chimera is the most common tool to visualize your density map. Here are some tricks
to viewing SPIDER files.

Open a SPIDER file
------------------

Chimera command line: open #0 spider:~/Desktop/enh_25_r7_05.ter

.. sourcecode:: sh
	
	chimera spider:~/Desktop/enh_25_r7_05.ter

View SPIDER Angle
-----------------

To see a specific orientation of your volume when using SPIDER angles,
the following commands may be used.

.. note::

	- SPIDER:  ZYZ rotating frame
	- CHIMERA: ZYX static frame

.. sourcecode:: c

	reset
	turn y theta coordinatesystem #0
	turn z phi coordinatesystem #0
	turn x 180







