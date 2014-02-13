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
		
		# Single image projects
		
		$ ara-project -i "<path-to-exposures>/mic*.mrc" --apix 1.3 --voltage 300 --cs 2.26 --particle-diameter 250 --mask-diameter 300 -w 20 --raw-reference-file emd1021.map
		
		# Movie mode projects
		
		$ ara-project -i "<path-to-exposures>/mic*.mrc" --apix 1.3 --voltage 300 --cs 2.26 --particle-diameter 250 --mask-diameter 300 -w 20 --gain-file <path-to-gain>/gain.mrc --raw-reference-file emd1021.map

2. Start the workflow

	.. sourcecode:: sh
	
		$ nohup sh run.sh  > local.log &

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
	For Arachnid>=0.1.3 the mask-diameter must match the mask diameter used in Relion.

5. Suggested AutoPicker parameters for various conditions/samples

	1. Crowded micrographs: --overlap-mult 0.8
	2. Very asymmetric particles (40S subunit of the ribosome) --disk-mult 0.2 
	3. Very few particles --threshold-minimum 10 (only works for Arachnid 0.1.3 or later)

6. Very Dirty Dataset - Use ara-vicer

	You must first run a short Relion Refinement, suggested on 4x decimated data. It does not have to run to the end, but 
	the longer you run it the better ara-vicer will work.
	
	To run, do the following
	
	.. sourcecode:: sh
		
		# Determine the good particles
		
		$ ara-vicer cluster/win/win_*.dat -a relion_it012_data.star -o output/view_0000000.dat -w8 -p cluster/data/params.dat 
		
	Note that this script writes out a relion selection file with the name view.star.

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

	$ ara-autopick --create-cfg auto.cfg

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

	$ ara-crop --create-cfg crop.cfg

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


