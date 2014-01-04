=======================
Reconstruction Protocol
=======================

This protocol describes single-particle reconstruction of a biological specimen (e.g. the ribosome) 
from a collection of electron micrographs.

.. contents:: 
	:depth: 1
	:local:
	:backlinks: none
	
Quick Start
===========

The following commands must be run in the same directory.

1. Generate a project workflow in current directory

	.. sourcecode:: sh
		
		$ sp-project ../mic*.mrc -r emd_1001.map -e spi -w 4 --apix 1.2 --voltage 300 --cs 2.26 --particle-diameter 220

2. Start the workflow

	.. sourcecode:: sh
	
		$ nohup sh run_local  > local.log &

3. Start manual micrograph/power spectra selection (Give some time for processing to start if not complete)

	.. sourcecode:: sh
	
		$ ara-screen
	
	By default, ara-screen displays the power spectra. You can view the micrographs by checking `Load Alternate`
	under the `Advanced Settings` on the `Loading` tab. Click the `Reload` Button to display the micrographs for
	the current set of images.
	
	.. note::
		
		The `Alternate Image` setting must be set to the micrograph file path for this work. If you used sp-project to perform your processing
		then ara-screen should automatically find the decimated micrograph.
	
	To view the particle selection, you need to check both `Load Alternate` and `Show Coords` under the `Advanced Settings` on 
	the `Loading` tab. 
	
	.. note::
		
		The `Coords` setting must be set to the coordinates file path for this work. If you used sp-project to perform your processing
		then ara-screen should automatically find the coordinates file unless you loaded ara-screen before AutoPicker started to run. If
		it did not find the  coordinates file you can set it by hand or close and reopen ara-screen in the same directory you last loaded it.

4. Decimated dataset for Relion Classification

	.. sourcecode:: sh
	
		ara-selrelion cluster/data/data.star -o cluster/data/data_dec02.star --downsample 2.0

Tips
====

1. Check the particle selection in ara-screen
	
	By default, ara-screen displays the power spectra

2. Check the reference
	
	Use Chimera to visualize the reference

3. Check the contrast inversion of the micrograph.
	
	It is assumed that your micrograph requires contrast inversion and the parameter `--is-film` 
	can keep the current contrast. You want light particles on a dark background.

4. Check normalization if you use Relion

	For Arachnid=0.1.2 the particle-diameter must match the mask diameter used in Relion.
	For Arachnid=0.1.3 the mask-diamter must match the mask diameter used in Relion.

5. Suggested AutoPicker parameters for various conditions/samples

	1. Crowded micrographs: --overlap-mult 0.8
	2. Very asymmetric particles (40S subunit of the ribosome) --disk-mult 0.2 
	3. Very few particles --threshold-minimum 10 (only works for Arachnid 0.1.3 or later)

6. Very Dirty Dataset - Use ara-autoclean

	You must first run a short Relion Refinement, suggested on 4x decimated data. It does not have to run to the end, but 
	the longer you run it the better ara-autoclean will work.
	
	To run, do the following
	
	.. sourcecode:: sh
		
		# Determine the good particles
		
		$ ara-autoclean cluster/win/win_*.dat -a relion_it012_data.star -o output/view_0000000.dat -w8 -p cluster/data/params.dat 
		
	Note that this script writes out a relion selection file with the name view.star.

Getting Started
===============

Now that you have finished collecting your data, you are ready to begin the image processing. At
this point you should have two things:

	#. A set of micrographs
	#. Information describing your data collection:
		- Pixel size, A
		- Electron energy, KeV
		- Spherical aberration, mm
		- Actual size of particle, angstroms

The default mode for project generation assumes that you will perform angular refinement and classification
with Relion. There are several additional options that are availabe using the :option:`--cluster-mode`.
	
Alternatively, the following subsections describe how to create and run a program from the command line.

Creating a project
------------------

The `sp-project` script generates an entire pySPIDER project including
	
	- Directories
	- Configuration files

A list of options (default configuration file) can be obtain by simply running
the program without arguments.

.. sourcecode:: sh
	
	$ sp-project
	ERROR:root:Option --input-files requires a value - found empty
	#  Program:	sp-project
	#  Version:	0.0.1
	
	#  Generate all the scripts and directories for a pySPIDER project
	#  
	#  $ sp-project micrograph_files* -o project-name -r raw-reference -e extension -w 4 --apix 1.2 --voltage 300 --cs 2.26 --particle-diameter 220 --scatter-doc ribosome
	#  

	
	input-files:                            #               (-i)    List of input filenames containing micrographs
	output:                                 #               (-o)    Output directory with project name
	raw-reference:                          #               (-r)    Raw reference volume - optional
	is-film:                         False   #		Set true if the micrographs were collected on film (or have been processed)
	apix:                           0.0     #       Pixel size, A
	voltage:                        0.0     #       Electron energy, KeV
	cs:                             0.0     #       Spherical aberration, mm
	particle-diameter:              0       #       Longest diameter of the particle, angstroms
	...

The values shown above (for brevity this is only a partial list of all available parameters) are all 
required for this script to run.

The values for each option can be set as follows:

.. sourcecode:: sh
	
	$ sp-project ../mic*.mrc -o ribosome_70s -r emd_1001.map -e spi -w 4 --apix 1.2 --voltage 300 --cs 2.26 --particle-diameter 220 --scatter-doc ribosome

Let's look at each parameter on the command line above.

The `../mic*.mrc` is a list of micrographs. The shell in most operating systems understands that `*` is a wildcard 
character that allows you to select all files in directory `../` that start with `mic` and end with `.mrc`. You do
not need to convert the micrographs to SPIDER format, that will be taken care of for you. In fact, the micrographs
are not converted at all, only the output particle projection windows are required to be in SPIDER format for
pySPIDER.

The `-o ribosome_70s` defines the name of the root output directory, which in this case is `ribosome_70s`. A set of
directories and configuration files/scripts will be created in this output directory (:ref:`see below <project-directory>`).

The `-r emd_1001.map` defines the raw reference volume. Ideally, this will be in MRC format with the pixel-size in the header. If not,
then you will need set the :option:`--curr-apix` parameter to set the proper pixel size.

The `-apix 1.2`, `--voltage 300`, `--cs 2.26`, and `--particle-diameter 220` microscope parameters that define the experiment.

The following are additional, recommended options.

The `-w 4` defines the number of cores to use for parallel processing.

The `--scatter-doc ribosome` will download a ribosome scattering file to 8A, otherwise you should specify an existing scattering file
or nothing.

.. note::

	When processing processed (i.e. already contrast inverted, e.g. film) micrographs `--is-film` should be added to the command above.
	In the configuration file, this should be `is-film: True`

.. _project-directory:

The command above will create a directory called `ribosome_70s` with the following structure:

.. sourcecode:: sh

	$ ls -R ribosome_70s
	ribosome_70s/:
	cluster  local run_local
	
	ribosome_70s/cluster:
	align.cfg  data  refine.cfg  refinement  win
	
	ribosome_70s/cluster/data:
	params.spi
	
	ribosome_70s/cluster/refinement:
	
	ribosome_70s/cluster/win:
	
	ribosome_70s/local:
	autopick.cfg  coords  crop.cfg  defocus.cfg  pow  reference.cfg
	
	ribosome_70s/local/coords:
	
	ribosome_70s/local/pow:

In the `ribosome_70s` directory, you will find two scripts: one to invoke all local scripts and one
to invoke the cluster scripts.

Running Scripts
---------------

To run all the local scripts in the proper order, use the following suggested command:

.. sourcecode:: sh

	$ cd ribosome_70s
	
	$ nohup sh run_local  > /dev/null &

.. note::
	
	All paths are setup relative to you executing a script from the project directory, e.g. `ribosome_70s`.

Improving the Data Treatment
============================

Screening
---------

Manually screening micrographs, power spectra and particle windows can all be done in `ara-view`.

.. note:: 
	
	Launch this program in the project directory and it will automatically find all necessary files.

.. sourcecode:: sh

	$ ara-screen

This program has several features:

  - Micrograph and power spectra screening can be done simutaneously
  - It can be used while collecting data, the `Load More` button will find more micrographs
  - Saving is only necessary when you are finished. It writes out SPIDER compatible selection files
  - Coordinates from AutoPicker can be displayed on the mcirographs

Additional processing
---------------------


Arachnid is geared toward automated data processing. Algorithms are currently under development to
handle each the of the steps below. Until such algorithms have been developed, it is recommended
that you use the SPIDER alternatives listed below. 

.. note:: 
	
	Arachnid was intended to be compatible with SPIDER batch files.
	
Manual CTF fitting
------------------

This can be done with `SPIDER's CTFMatch <http://www.wadsworth.org/spider_doc/spire/doc/guitools/ctfmatch/ctfmatch.html>`_. CTFMatch
will write out a new defocus file

.. note::
	
	It is recommended that you rename the current defocus file first, then save the new defocus file 
	with the original name of the current defocus file.

Classification
--------------

#. Supervised Classification
	
	See: http://www.wadsworth.org/spider_doc/spider/docs/techs/supclass/supclass.htm

Chimera Tricks
==============

Chimera is the most common tool to visualize your density map. Here are some tricks
to viewing SPIDER files.

Open a SPIDER file
------------------

Chimera command line: open #0 spider:~/Desktop/enh_25_r7_05.ter

.. sourcecode:: sh
	
	chimera spider:~/Desktop/enh_25_r7_05.ter

Choose a SPIDER Viewing Angle
-----------------------------

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

Running Isolated Scripts
========================

This section covers running Arachnid scripts in isolation, i.e. when you only want to use Arachnid for one
procedure in the single-particle reconstruction workflow.

Particle Selection
------------------

1. Create a config file

.. sourcecode:: sh

	$ ara-autopick > auto.cfg

2. Edit config file

.. sourcecode:: sh

	$ vi auto.cfg
	
	# - or -
	
	$ kwrite auto.cfg

	input-files: Micrographs/mic_*.spi
	output:	coords/sndc_0000.spi
	param-file: params.spi
	bin-factor: 2.0
	worker-count: 4
	invert: False 	# Set True for unprocessed CCD micrographs

3. Run using config file

.. sourcecode:: sh
	
	$ ara-autopick -c auto.cfg

Particle Windowing
------------------

1. Create a config file

.. sourcecode:: sh

	$ ara-crop > crop.cfg

2. Edit config file

.. sourcecode:: sh

	$ vi crop.cfg
	
	# - or -
	
	$ kwrite crop.cfg

	input-files: Micrographs/mic_*.spi
	output:	win/win_0000.spi
	coordinate-file: coords/sndc_0000.spi
	param-file: params.spi
	bin-factor: 1.0
	worker-count: 4		# Set based on number of available cores and memory limitations
	invert: False 		# Set True for unprocessed CCD micrographs

3. Run using config file

.. sourcecode:: sh
	
	$ ara-crop -c auto.cfg

Creating Relion Selection File
------------------------------

.. sourcecode:: sh
	
	$ ara-selrelion -i win/win_* -o relion_input.star -p params.dat -d defocus.dat


