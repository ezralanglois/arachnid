=======
Manual
=======

This manual covers the usage of every available script. It provides a list of
options and corresponding example usage.

.. contents:: 
	:depth: 1
	:local:
	:backlinks: none
	
Scripts 
=======

.. toctree::
	:maxdepth: 2
	
	Applications <app/manual>
	Utilities <util/manual>
	pySPIDER <pyspider/manual>

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
	
	$ ap-autopick
	2012-08-05 11:31:06,039 ERROR Requires at least one file
	#  Program:	ap-autopick
	#  Version:	1.1.8
	#  Usage: ap-autopick inputfile1 inputfile2 --option-name value ...
	
	#  This program find particles with reference-free Gaussian homomorphic cross-correlation.
	#  
	#  $ ap-autopick input-stack.spi -o coords.dat
	#  
	
	input-files:			#		(-i)	Input files in addition to those placed on the command line
	output:				#		(-o)	Path and name of the output file

In the above example, the configuration file prints a header describing:
	
	#. Name of the program that generated the configuration file
	#. Version of the program that generated the configuration file
	#. Basic usage of the program from the console

Any parameter listed in the configuration file can be used on the command line by adding `--` to the
beginning.

.. sourcecode:: sh
	
	$ ap-autopick --input-files mic_0001.spi,mic_0002.spi,mic_0003.spi --output sndc_0000.spi

Some parameters have the option to use a short flag, e.g. `-o`.

.. sourcecode:: sh
	
	$ ap-autopick -i mic_0001.spi,mic_0002.spi,mic_0003.spi -o sndc_0000.spi

Finally, for input file only, you have the option of no flag. In this case only, multiple files must be 
separated by spaces (whereas if you use `-i` or `--input-files`, they must be separated by commas).

.. sourcecode:: sh
	
	$ ap-autopick mic_0001.spi mic_0002.spi mic_0003.spi -o sndc_0000.spi

Wild cards may be used for input files, however if `-i` or `--input-files`, then the input filename
show be quoted.

.. sourcecode:: sh
	
	$ ap-autopick mic_*.spi -o sndc_0000.spi
	$ ap-autopick -i "mic_*.spi" -o sndc_0000.spi
	
.. note::
	
	Using `ap-autopick -i "mic_*.spi" -o sndc_0000.spi` will allow you to exceed the terminal limit on number
	of input files.
	
A configruation file may be created in a number of ways. First, it can be created by redirecting the output
of a script with no parameters to a filename. Note that this will always print an error to the console, which
can be disregarded.

.. sourcecode:: sh
    
    $ ap-autopick > autopick.cfg		# Create configuration file
    2012-08-05 11:31:06,039 ERROR Requires at least one file
    
    $ vi autopick.cfg					# Edit configuration file
	#  Program:	ap-autopick
	#  Version:	1.1.8
	#  Usage: ap-autopick inputfile1 inputfile2 --option-name value ...
	
	#  This program find particles with reference-free Gaussian homomorphic cross-correlation.
	#  
	#  $ ap-autopick input-stack.spi -o coords.dat
	#  
	
	input-files:			#		(-i)	Input files in addition to those placed on the command line
	output:				#		(-o)	Path and name of the output file

	...

A configuration file may also be created with the `--create-cfg` option. You may give
it a value like `1` and it will write to the standard output similar to the previous
example but without an error message. You can also give it a filename directly.

.. sourcecode:: sh
    
    $ ap-autopick --create-cfg 1 > autopick.cfg 	# Create configuration file
    $ ap-autopick --create-cfg autopick.cfg		# Create configuration file

The `--create-cfg` option can also be used to create a copy of a configuration file. This
is particulary useful when you want to change certain values from the command line.

.. sourcecode:: sh
    
    $ ap-autopick -c autopick.cfg --create-cfg autopick_new.cfg -i mic_0001.spi

Local Machine
-------------

Running an Arachnid script with a configuration file is fairly simple, it only requires
you specify the name and path of the configuration file.

.. sourcecode:: sh
    
    $ ap-autopick -c autopick.cfg

The `-c` option is short for `--config-file` and takes the filename of the configuration
file as input.

If you do not specify a log file and wish to keep this information in a file, then you
can redirect the output of the script to a file.

.. sourcecode:: sh
    
    $ ap-autopick -c autopick.cfg 2> autopick.log 	# Bash Shell
    $ ap-autopick -c autopick.cfg >& autopick.log 	# C-Shell

Also, you may want to run the script in the background.
    
.. sourcecode:: sh
    
    $ ap-autopick -c autopick.cfg 2> autopick.log &	# Bash Shell
    $ ap-autopick -c autopick.cfg >& autopick.log &	# C-Shell

If you are running the command from a remote machine and may lose the connection, then
you may add the `nohup` command to keep the program running even if the connection is severed.
    
.. sourcecode:: sh
    
    $ nohup ap-autopick -c autopick.cfg 2> autopick.log &	# Bash Shell
    $ nohup ap-autopick -c autopick.cfg >& autopick.log &	# C-Shell

Long, difficult commands or simple scripts may be placed in the configuration file. Each
command or command in the script must be proceeded by a space. A space at the start of the 
line is considered a comment character in the configuration file and anything following 
a space at the start of a line is ignored. You should also end your command sequence with
`exit 0` to ensure no superflous error messages are created.

.. sourcecode:: sh
    
    $ vi autopick.cfg					# Edit configuration file
	#  Program:	ap-autopick
	#  Version:	1.1.8
	#  Usage: ap-autopick inputfile1 inputfile2 --option-name value ...
	
	#  This program find particles with reference-free Gaussian homomorphic cross-correlation.
	#  
	#  $ ap-autopick input-stack.spi -o coords.dat
	#  
	
	# Note the space proceeding each command, these lines are ignored when the config file is read
	 nohup ap-autopick -c autopick.cfg 2> autopick.log &
	 exit 0
	
	input-files:			#		(-i)	Input files in addition to those placed on the command line
	output:				#		(-o)	Path and name of the output file
	
	...

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

Below is an example showing how to run `ap-autopick` on a cluster of 10 nodes where `machinefile` contains a list of the 
nodes that the script is allowed to use.

.. sourcecode:: sh

	nohup mpiexec -stdin none -n 10 -machinefile machinefile ap-autopick -c autopick.cfg --use-MPI < /dev/null > autopick.log &


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

	$ ap-autopick > auto.cfg
	
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

Logging in arachnid is both versatile and redundant to give the user the best of all worlds. The most basic control the user
has is the level of logging, i.e. the amount of information printed by the program.

Most users should use `--log-level` 3 or info, which is the default. This prints some valuble information along with
warnings and errors. Additional information can be obtained with `--log-level` 4 or debug. See examples below:

.. sourcecode: sh

	$ ap-autopick -c autopick -v4
	
	$ ap-autopick -c autopick -vdebug
	
	$ ap-autopick -c autopick --log-level 4
	
	$ ap-autopick -c autopick --log-level debug

Using `--log-level` 2 or warn causes the script to only print warnings and errors.

Using `--log-level` 1 or error causes the script to only print only errors.

Terminal
--------

By default, an Arachnid script will log messages to STDERR, which means catching log messages can be done as follows:

.. sourcecode:: sh
	
	$ ap-autopick -c autopick 2> log.txt # For Bash (bash) or Bourne Shell (sh) or Korn Shell (ksh)
	#
	# Or
	#
	$ ap-autopick -c autopick >& log.txt # For C-shell (csh)

Log File
--------

You also have the option of specifying a log file (however, messages will still be printed to STDERR).

.. sourcecode:: sh
	
	$ ap-autopick -c autopick --log-file log.txt

You can log only to the log file as follows:

.. sourcecode:: sh
	
	$ ap-autopick -c autopick --log-file log.txt --disable-stderr

An added benefit of using `--log-file` is that old log files (with the same name) will be backed up in a zip
archive (of the same name but with a '.zip' extension). 

Crash Report
------------

Also, note that a crash report is generated regardless of what logging mode you choose. This file is
called .$PROGRAM_NAME.crash_report. For example, for ap-autopick, it will be called 
`.ap-autopick.crash_report`.

This file contains any exceptions that are thrown during the execution of the script and is useful for reporting
bugs. Please attach this file when you submit an issue.

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

The following options are shared by all scripts and are organized by their utlity.

Critical Options
++++++++++++++++

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of filenames for the input micrographs. If you use the parameters `-i` or `--inputfiles` they must be
    comma separated (no spaces). If you do not use a flag, then separate by spaces. For a very large number of files
    (>5000) use `-i "filename*"`
    
.. option:: -o <FILENAME>, --output <FILENAME>
    
    Filename and path to output file with correct number of digits (e.g. vol_0000.spi)

Logging Options
+++++++++++++++

.. option:: -v <CHOICE>, --log-level <CHOICE>
    
    Set logging level application wide: 'critical', 'error', 'warning', 'info', 'debug' or 0-4

.. option:: --log-file <FILENAME>
    
    Set file to log messages

.. option:: --log-config <FILENAME>
    
    File containing the configuration of the application logging

.. option:: --disable_stderr <BOOL>
    
    If true, output will only be written to the given log file
    
Other Options
+++++++++++++

.. option:: --create-cfg <STRING>
	
	Create a configuration file, if STRING is a filename, then write to that file, 
	otherwise write to the standard output

.. option:: --prog-version <STRING>
	
	Version of the program that created the configuration file. Chaning this value
	will load the specified version of the program at run-time.

.. _file-proc-options:

File Processor
--------------

A subset of scripts support individual file processing. The following options are available for
these scripts.

.. option:: -l <int>, --id-len <int>
    
    Set the expected length of the document file ID

.. option:: -b <str>, --restart-file <str>
    
    Set the restart file to keep track of which processes have finished. If this is empty, then a restart file is 
    written to .restart.$script_name You must set this to restart (otherwise it will overwrite the automatic restart file).

.. option:: -w <int>, --work-count <int>
    
     Set the number of micrographs to process in parallel (keep in mind memory and processor restrictions)

.. _param-options:

Cryo-EM Experiments
-------------------

The following options are supported by most of the scripts. They define key parameters that define
a cryo-EM experiment.

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Filename for SPIDER parameter file describing a Cryo-EM experiment

.. option:: --bin-factor <FLOAT>
    
    Decimatation factor for the script: changes size of images, coordinates, parameters such as pixel_size or window unless otherwise specified


.. _mpi-options:

MPI-Enabled Scripts
-------------------

These options are shared by all mpi-enabled scripts. For pySPIDER scripts, each of the directories is 
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

.. option:: --use-MPI <BOOL>
	
	Set this flag True when using mpirun or mpiexec

.. option:: --shared-scratch <FILENAME>
	
	File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)

.. option:: --home-prefix <FILENAME>
	
	File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)

.. option:: --local-scratch <FILENAME>

	File directory on local node to copy files (optional but recommended for MPI jobs)

.. option:: --local-temp <FILENAME>

	File directory on local node to setup a soft link to `home-prefix` (optional but recommended for MPI jobs)

.. _spider-options:

|spi| Scripts
-------------

These options are shared pyspider scripts; those prefixed with `spi-`.

.. option:: --spider-path <FILENAME>
    
    File path to spider executable
    
.. option:: --data-ext <str>
    
    Extension of spider data files
    
.. option:: --thread-count <int>
    
    Number of threads per machine
    
.. option:: --enable-results <BOOL>
    
     If set true, print results file to terminal

