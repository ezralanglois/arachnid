================
pySpider Manual
================

This manual covers the usage of every available |spi| batch file. It provides a list of
options and corresponding example usage. See :doc:`../manual` for more details
on how to run each script.

.. contents:: 
	:depth: 1
	:local:
	:backlinks: none
	
Project Creation
================

.. automodule:: arachnid.pyspider.project
	:noindex:
    
Defocus Estimation
==================

.. automodule:: arachnid.pyspider.defocus
	:noindex:

Reference Creation
==================

.. automodule:: arachnid.pyspider.reference
	:noindex:

Reference-based Alignment
=========================

.. automodule:: arachnid.pyspider.align
	:noindex:
	
Empty Alignment
===============

.. automodule:: arachnid.pyspider.create_align
	:noindex:

Refinement
==========

.. automodule:: arachnid.pyspider.refine
	:noindex:

Reconstruction
===============

.. automodule:: arachnid.pyspider.reconstruct
	:noindex:

Alignment Classification
========================

.. automodule:: arachnid.pyspider.classify
	:noindex:

Volume Pre-processing (for Refinement)
======================================

.. automodule:: arachnid.pyspider.prepare_volume
	:noindex:

Amplitude Enhance a Volume
==========================

.. automodule:: arachnid.pyspider.enhance_volume
	:noindex:

Filter a Volume
===============

.. automodule:: arachnid.pyspider.filter_volume
	:noindex:

Mask a Volume
=============

.. automodule:: arachnid.pyspider.mask_volume
	:noindex:

Estimate Resolution
===================

.. automodule:: arachnid.pyspider.resolution
	:noindex:


TODO
====
 
 #. Test logging code!
 #. Test new reconstruct pruning code
 #. Test defocus decimation
 #. Test decimation in refinement
 #. Print reference used in refinement
 #. Add pixel size to alignment and undecimate alignment file during align/refine
 #. Write out class average stack at the end of alignment
 #. Version control configuration - with git tag!
 #. Chimera Session - edit existing session
 #. Store pixel size in map (stacks)
 #. Brix conversion
 #. Amplitude Enhanced
 #. MPI Autopicker and defocus (auto skip zeros)
 #. One touch refinement
 #. Script to generate all configurations files, setup directory structures, and default filenames (with default values for full SPIDER refinement)
 #. Add time estimation for each alignment
 #. Get logging to work properly (including spider results file)
 #. Special MPI gui for MPI scripts that sets command at top of configuration file
 #. AutoGUI
 #. ImageViewer
 #. ImageSelector
 #. Plotting
 #. Documentation of how to setup MPI on our cluster
 #. Fix refinement bug
 #. MPI FFTW http://www.fftw.org/doc/2d-MPI-example.html#g_t2d-MPI-example
