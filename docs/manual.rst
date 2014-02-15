=======
Manual
=======

This manual covers the usage of every available script. It provides a list of
options and corresponding example usage.

.. note::

	This manual is intended for more advanced users, first time users
	should see :doc:`reconstruction`

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

This section covers the basic usage of any Arachnid script from the console.

Configuration Files
-------------------

To get help or a list of options from the command-line, simply invoke the Arachnid script of
interest on the command line. This will print an error, e.g. `Requires at least one file` and
a configuration file (see below).

.. sourcecode:: sh
	
	$ ara-autopick -h
	#  Program:	ara-autopick
	#  Version:	0.1.4_158_gc3e4798
	#  URL:	http://www.arachnid.us/docs/api_generated/arachnid.app.autopick.html
	#
	#  CITE:	http://www.arachnid.us/CITE.html
	
	# Automated particle selection (AutoPicker)
	# 
	# $ ls input-stack_*.spi
	# input-stack_0001.spi input-stack_0002.spi input-stack_0003.spi
	# 
	# Example: Unprocessed film micrograph
	# 
	# $ ara-autopick input-stack_*.spi -o coords_00001.dat -r 110
	# 
	# Example: Unprocessed CCD micrograph
	# 
	# $ ara-autopick input-stack_*.spi -o coords_00001.dat -r 110 --invert
	# 
	   ara-autopick -c $PWD/$0 $@
	   exit $?
	
	
	
	
	
	#  Options that must be set to run the program
	input-files:				#		(-i,--micrograph-files)	List of filenames for the input micrographs, e.g. mic_*.mrc
	output:					    #		(-o,--coordinate-file)	Output filename for the coordinate file with correct number of digits (e.g. sndc_0000.spi)


In the above example, the configuration file prints a header describing:
	
	#. Name of the program that generated the configuration file
	#. Version of the program that generated the configuration file
	#. Location of documentation for program
	#. Basic usage of the program from the console
	#. Code to run the script with: `sh name-of-script.cfg`
	
.. warning::
	
	Spaces at the start of a line are treated as comments!

Any parameter listed in the configuration file can be used on the command line by adding `--` to the
beginning.

.. sourcecode:: sh
	
	$ ara-autopick --input-files mic_0001.spi,mic_0002.spi,mic_0003.spi --output sndc_0000.spi

Some parameters have the option to use a short flag, e.g. `-o`.

.. sourcecode:: sh
	
	$ ara-autopick -i mic_0001.spi,mic_0002.spi,mic_0003.spi -o sndc_0000.spi

Finally, for input file only, you have the option of no flag. In this case only, multiple files **must** be 
separated by spaces (whereas if you use `-i` or `--input-files`, they **must** be separated by commas).

.. sourcecode:: sh
	
	$ ara-autopick mic_0001.spi mic_0002.spi mic_0003.spi -o sndc_0000.spi

Wild cards may be used for input files, however if `-i` or `--input-files`, then the input filename
show be quoted.

.. sourcecode:: sh
	
	$ ara-autopick mic_*.spi -o sndc_0000.spi
	$ ara-autopick -i "mic_*.spi" -o sndc_0000.spi
	
.. note::
	
	Using `ara-autopick -i "mic_*.spi" -o sndc_0000.spi` will allow you to exceed the terminal limit on number
	of input files.
	
A configruation file may be created in a number of ways. First, it can be created by redirecting the output
of a script with no parameters to a filename. Note that this will always print an error to the console, which
can be disregarded.

.. sourcecode:: sh
	
    # Create configuration file
    
    $ ara-autopick --create-cfg autopick.cfg
    
    
    # Edit configuration file
    
    $ vi autopick.cfg					
	#  Program:	ara-autopick
	#  Version:	0.1.4_158_gc3e4798
	#  URL:	http://www.arachnid.us/docs/api_generated/arachnid.app.autopick.html

	...

The `--create-cfg` option can also be used to create a copy of a configuration file. This
is particulary useful when you want to change certain values from the command line.

.. sourcecode:: sh
    
    $ ara-autopick -c autopick.cfg --create-cfg autopick_new.cfg -i mic_0001.spi

Local Machine
-------------

Running an Arachnid script with a configuration file is fairly simple, it only requires
you specify the name and path of the configuration file.

.. sourcecode:: sh
    
    $ ara-autopick -c autopick.cfg

The `-c` option is short for `--config-file` and takes the filename of the configuration
file as input.

If you do not specify a log file and wish to keep this information in a file, then you
can redirect the output of the script to a file.

.. sourcecode:: sh
    
    $ ara-autopick -c autopick.cfg 2> autopick.log 	# Bash Shell
    $ ara-autopick -c autopick.cfg >& autopick.log 	# C-Shell

Also, you may want to run the script in the background.
    
.. sourcecode:: sh
    
    $ ara-autopick -c autopick.cfg 2> autopick.log &	# Bash Shell
    $ ara-autopick -c autopick.cfg >& autopick.log &	# C-Shell

If you are running the command from a remote machine and may lose the connection, then
you may add the `nohup` command to keep the program running even if the connection is severed.
    
.. sourcecode:: sh
    
    $ nohup ara-autopick -c autopick.cfg 2> autopick.log &	# Bash Shell
    $ nohup ara-autopick -c autopick.cfg >& autopick.log &	# C-Shell

Long, difficult commands or simple scripts may be placed in the configuration file. Each
command or command in the script must be proceeded by a space. A space at the start of the 
line is considered a comment character in the configuration file and anything following 
a space at the start of a line is ignored. You should also end your command sequence with
`exit 0` to ensure no superflous error messages are created.

.. sourcecode:: sh
    
    $ vi autopick.cfg					# Edit configuration file
	#  Program:	ara-autopick
	#  Version:	1.1.8
	#  Usage: ara-autopick inputfile1 inputfile2 --option-name value ...
	
	#  This program find particles with reference-free Gaussian homomorphic cross-correlation.
	#  
	#  $ ara-autopick input-stack.spi -o coords.dat
	#  
	
	# Note the space proceeding each command, these lines are ignored when the config file is read
	 nohup ara-autopick -c autopick.cfg 2> autopick.log &
	 exit 0
	
	input-files:			#		(-i)	Input files in addition to those placed on the command line
	output:				#		(-o)	Path and name of the output file
	
	...

.. warning::
	
	Remember, spaces at the start of a line are treated as comments!

After adding your commands, you can now execute the configuration file like a script.

.. sourcecode:: sh
    
    $ sh autopick.cfg


Cluster
-------

Many but not all scripts in Arachnid support the message passing interface (MPI) allowing execution on most clusters. Some
clusters have a scheduling system and require a special script to launch a job. You will need to contact your local
administrator to find out details about how to setup your job on these types of systems.

An Arachnid script that supports MPI will have the `use-MPI` option in the configuration file. Setting this option `True` or
adding it to the command line will enable the MPI-ready version of the code.

Below is an example showing how to run `ara-autopick` on a cluster of 10 nodes where `machinefile` contains a list of the 
nodes that the script is allowed to use.

.. sourcecode:: sh

	nohup mpiexec -stdin none -n 10 -machinefile machinefile ara-autopick -c autopick.cfg --use-MPI < /dev/null > autopick.log &


.. _version_control:

Version Control
===============

Arachnid employs two types of version control:

#. Version control on the script-level
	
	Every configuration file you create maintains the version of arachnid used to create it. If you install a new version and
	the old version is kept, then running the script over the configuration file will cause the code to automatically revert
	to the version that created the script.

#. Version control on the config-level

	If you run the same script several times and keep the same output file name, then the script will maintain every change you
	make in a version control configuration file. If you forgot which options you set to get a particular result, this config
	file can help retrace your steps.

Versioning the Script
---------------------

Every configuration file you create maintains the version of arachnid used to create it. It does this
using the `prog-version` option (see below).

.. sourcecode:: sh

	$ ara-autopick > auto.cfg
	
	$ more auto.cfg
	
	...
	good-output:				#	Output coordindates for the good particles for performance benchmark
	use-MPI:			False	#	Set this flag True when using mpirun or mpiexec
	prog-version:			0.0.1	#	Select version of the program (set `latest` to use the lastest version`)
	...

If you run a newer version of the same script (and the older version was not removed) then the script will automatically
revert back to the previous version giving you a warning message. 

If you want to use the latest version, then you can set `prog-version: latest`.

Versioning the Config File
--------------------------

If you run a script, it will automatically create an extra file based on the output file you 
specify, e.g. output: folder/name.ext ---> folder/cfg_name.cfg.

As long as the output file does not change, this file will record a "version-controlled" configuration 
file. Every time you run the program, a date following by the parameters you changed will be 
written to this file. This file in itself is a valid configuration file. If you want go back to a previous
experiment, then you need only make a copy of this file and remove all options below a certain date.

.. sourcecode:: sh
	
	$ more folder/cfg_name.cfg
	# 2012-3-30 5:30:23
	input-files: file1
	boost:		1002
	
	# 2012-3-30 6:30:23
	boost:		1003
	
	# 2012-4-30 2:31:23
	boost:		900

If I use the above file as a configuration file, then the run will be the same as was performed on 2012-4-30 at 2:31:23.

.. sourcecode:: sh
	
	$ more folder/cfg_name.cfg
	# 2012-3-30 5:30:23
	input-files: file1
	boost:		1002
	
	# 2012-3-30 6:30:23
	boost:		1003

If I use the above truncated file as a configuration file, then the run will be the same as was performed on 2012-3-30 at 6:30:23.

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


