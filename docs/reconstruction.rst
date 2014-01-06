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

#. Run the following to load the Graphical User Interface.

	.. sourcecode:: sh
	
		$ ara-control
	
#. This command opens the following window.
	
	.. image:: images/wizard_000.png
		:scale: 20%
		:target: content_

	.. _content:

	.. container:: content
		
		.. image:: images/wizard_000.png

#. You will then be asked whether you wish to automatically import or manually enter certain information describing your experiment.

	.. image:: images/wizard_010.png
		:scale: 20%
		:target: content02_

	.. _content02:

	.. container:: content
		
		.. image:: images/wizard_010.png

#. Assuming you choose to import from the leginon database and **this is your first time**, then you will see the following page:

	This page asks you for the following information, which you will need to obtain from
	your systems administrator:
	
		- Hostname or IP Address for Leginon Primary Database followed by name of the database
		- Hostname or IP Address of Leginon Project Database followed by name of the database
		- Leginon Credentials (You should know this)
		- Database Credentials

	.. image:: images/wizard_022.png
		:scale: 20%
		:target: content03_

	.. _content03:

	.. container:: content
		
		.. image:: images/wizard_022.png

#. Once you have successfully logged into the Leginon Database, then you should see the following page:

	If this is not your first time, then you should see this page first.
	
	Note that the last session for which you collected data should be displayed. You can
	view more by increasing the number shown and then clicking the refresh button.

	.. image:: images/wizard_021.png
		:scale: 20%
		:target: content04_

	.. _content04:

	.. container:: content
		
		.. image:: images/wizard_021.png

#. You must select at least one session to process before you can continue.

	.. image:: images/wizard_020.png
		:scale: 20%
		:target: content05_

	.. _content05:

	.. container:: content
		
		.. image:: images/wizard_020.png

Next

Tips
====

1. Check the particle selection in ara-screen
	
	By default, ara-screen displays the power spectra

2. Check the reference
	
	Use Chimera to visualize the reference

3. Check the contrast inversion of the micrograph.
	
	It is assumed that your micrograph requires contrast inversion and the parameter `--is-film` 
	can keep the current contrast. You want light particles on a dark background.

4. Check normalization when preparing the data for Relion

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

Alternative
===========

You may alternatively wish to run everything from the command line. This is covered in the 
:ref:`Command-line Protocol <reconstruction_cmdline>`.

