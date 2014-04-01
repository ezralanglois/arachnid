========================
Post Processing Protocol
========================

This protocol describes post-processing of the single-particle reconstruction run
by either Relion or SPIDER.

Relion and SPIDER
=================

The following describes post-processing of either SPIDER or Relion alignment
files.

Angular Distribution
--------------------

The distribution of the projections in 3D space is an important factor in the quality of
a reconstruction. It indicates areas on the reconstruction that may be artifactual or just
low-resolution due to missing projections.

The `ara-coverage` script, by default, maps a visual histogram of projection distribution
on a 2D projection of a sphere.

.. sourcecode:: sh

	$ ara-coverage relion-it001_data.star -o coverage.png

Relion-Only
============

The following describes post-processing of Relion alignment files.

Class Selection
---------------

After classification, it is usually necessary to extract out a single class of particles for
further refinement. This is can be done as follows:

.. sourcecode:: sh

	$ ara-selrelion relion-it001_data.star -o relion-it001_data_class_1_2_5.star -s1,2,5
	

Sometimes, it is usually necessary to recombine several classes into a single star file for 
refinement. This is can be done as follows:

.. sourcecode:: sh

	$ ara-selrelion relion-it001_data.star -o relion-it001_data_class_1_2_5.star -s1,2,5
	
.. note::
	
	Running autorefine with the above selection files will start with the angles found by
	classification. To begin from scratch include the `--restart` option.


Relion Movie-mode Alignment
---------------------------

This section assumes you have already performed classification/refinement with Relion and
obtained a structure that is better than 6-angstroms. It is also assumed that you:

#. Performed frame alignment on the micrographs with Arachnid or do not require frame alignment.
#. Or, know how to convert the translations from the software you used to Arachnid
#. Or, do not require translations for the micrographs

Step 1: Generate a selection file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Normally, you have performed some type of classification for your data and only wish to crop
frames for a subset, which you subsequently refined to better than 6-angstroms. Even if you
want to crop all the data, performing this step will make things easier in the future.

The following command will generate a particle selection file by micrograph in the SPIDER
document format.

.. sourcecode:: sh

	$ ara-selrelion relion-it001_data.star -o good/good_000000.dat

Step 2: Crop the frames
~~~~~~~~~~~~~~~~~~~~~~~


.. sourcecode:: sh

	$ ara-crop local/frames/mic_* -g good/good_000000.dat -o win/win_000000.dat -w16

Step 3: Generate a Relion Selection File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. sourcecode:: sh

	$ ara-selrelion relion-it001_data.star --frame-stack-file "win/frame_*_win_000000.dat" --reindex-file good/good_000000.dat -o relion-it001_data_frames.star

.. note:: 
	
	In the config file, the quotes are unnecessary, e.g.:
	
	frame-stack-file: win/frame_*_win_000000.dat
	

Step 4: Test the Relion Selection File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example assumes the following:

#. You have 16 cores available on your system.
#. The pixel size used for CTF estimation is 1.5.

.. sourcecode:: sh

	$ ara-reconstruct relion-it001_data_frames.star -o raw.dat -t16 --apix 1.5

Chimera
=======

Ctrl to select segment before merging.

segment exportmask sel savePath mask.mrc


