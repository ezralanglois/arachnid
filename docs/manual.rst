=======
Manual
=======

This manual covers the usage of every available script. It provides a list of
options and corresponding example usage.

Tutorials
=========

Most users may be interesting in one of the following tutorials:

.. list-table::
   :class: contentstable

   * - :doc:`Beginner Tutorial <reconstruction>`

       Single-particle reconstruction with the GUI

     - :doc:`Advanced Tutorial <reconstruction_cmdline>`

       Single-particle reconstruction from the command line

   * - :doc:`Post-processing Tutorial <reconstruction_post_cmdline>` 

       Post-processing the data after classification and refinement

     - `Latest Tips and Tricks <http://blog.arachnid.us/>`_

       Latest information about the daily builds, upcoming versions and other useful tidbits

Scripts
=======

Applications
------------

.. currentmodule:: arachnid.app

.. autosummary::
	:nosignatures:
	:template: man_module.rst
	:toctree: man_generated/
	
	align_frames
	autopick
	lfcpick
	reconstruct
	vicer

Utilities
---------

.. currentmodule:: arachnid.util

.. autosummary::
	:nosignatures:
	:template: man_module.rst
	:toctree: man_generated/
	
	bench
	coverage
	crop
	enumerate_filenames
	image_info
	project
    relion_selection
    
pySPIDER
--------

.. currentmodule:: arachnid.pyspider

.. autosummary::
	:nosignatures:
	:template: man_module.rst
	:toctree: man_generated/
    
    align
    autorefine
    classify
    create_align
    defocus
    enhance_volume
    filter_volume
   	mask_volume
    prepare_volume
    reconstruct
    reference
    refine
    resolution

.. _running-scripts:

Running the Scripts
===================

.. include:: /arachnid/core/app/settings.py
	:start-after: beg-usage
	:end-before: end-usage

Logging
=======

.. include:: /arachnid/core/app/tracing.py
	:start-after: beg-usage
	:end-before: end-usage

.. _common-options:

Common Options
==============

This section lists common options shared:

	- :ref:`shared-options`
	- :ref:`mpi-options`
	- :ref:`spider-options`

.. _shared-options:

All Scripts
------------

The following options are shared by all app/util scripts and are organized by their utility.

.. note::
	
	These options are generally not supported by GUI-only
	applications.

Critical Options
++++++++++++++++

.. include:: /arachnid/core/app/program.py
	:start-after: beg-convention-options
	:end-before: end-convention-options

Logging Options
+++++++++++++++

.. include:: /arachnid/core/app/tracing.py
	:start-after: beg-options
	:end-before: end-options
    
Other Options
+++++++++++++

.. include:: /arachnid/core/app/program.py
	:start-after: beg-program-options
	:end-before: end-program-options

.. _file-proc-options:

File Processor
--------------

A subset of scripts support individual file processing. The following options are available for
these scripts.

.. program:: file-processor

.. include:: /arachnid/core/app/file_processor.py
	:start-after: beg-options
	:end-before: end-options

.. _param-options:

Cryo-EM Experiments
-------------------

.. warning:: 

	These parameters will be removed in the next version!

The following options are supported by most of the scripts. They define key parameters that define
a cryo-EM experiment.

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Filename for SPIDER parameter file describing a Cryo-EM experiment

.. option:: --bin-factor <FLOAT>
    
    Decimatation factor for the script: changes size of images, coordinates, parameters such as pixel_size or window unless otherwise specified

.. _mpi-options:

MPI-Enabled Scripts
-------------------

These options are shared by all MPI-enabled scripts. For pySPIDER scripts, each of the directories is 
extermely important for proper function:

 - The :option:`--shared-scratch` option is important for native SPIDER reconstruction engines; it is used 
   to communicate the separate aligned stacks to the node responsible for the reconstruction. 
 
 - The :option:`--local-scratch` option is the location where partitions of the entire dataset are written. If
   this is on the local harddrive of the computing node (rather than a mounted network drive) then you will see
   a substantial performance increase.
   
 - The :option:`--local-scratch` and `--local-temp` options work together to shorten every filename. The
   `--local-temp` directory will ideally be `/tmp`. This is important because SPIDER can only handle file names
   less than 80 characters long. With these two parameters set proplerly, a soft link is created to automatically
   shorted all the filenames in the script.

.. program:: mpi-enabled

.. include:: /arachnid/core/app/program.py
	:start-after: beg-mpi-options
	:end-before: end-mpi-options

OpenMP-Enabled Scripts
----------------------

This option is shared by all non-pySPIDER OpenMP-enabled scripts.

.. program:: openmp-enabled

.. include:: /arachnid/core/app/program.py
	:start-after: beg-openmp-options
	:end-before: end-openmp-options

.. _spider-options:

|spi| Scripts
-------------

These options are shared pyspider scripts; those prefixed with `spi-`.


.. program:: pyspider

.. include:: /arachnid/core/spider/spider.py
	:start-after: beg-options
	:end-before: end-options


