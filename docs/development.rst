===================
Developer's Guide
===================

.. contents:: 
	:depth: 1
	:local:
	:backlinks: none
	
	
------------
Custom Code
------------

Arachnid supports two types of scripting: applications and fast prototyping. Applications are scripts
that are designed for a user and to fit on the reconstruction workflow. Fast prototyping, however, is
a way to test new ideas, perform a simple task quickly or customize objects such as plots.

The Snippets below cover a set of examples for writing fast-prototype code. These
examples range from unstacking a SPIDER image stack to customizing the FSC plot.

.. currentmodule:: arachnid.snippets

.. autosummary::
    :nosignatures:
    :toctree: api_generated/
    :template: api_module.rst
    
    plot_fsc
    unstack
    filter_volume
    mask_volume
    reconstruct
    reconstruct3
    postprocess_volume
    estimate_resolution
    center_images
    plot_angles
    scale_align
    corrupt_particles
    phase_flip
 
-------------------------------
Application Programs Interface
-------------------------------

See :doc:`api` for a full list of available modules

-----------
Development
-----------

.. contents:: 
	:depth: 1
	:local:
	:backlinks: none

Getting Source
==============

To get the latest version of the source, use the following `git <http://git-scm.com/documentation>`_ command:

.. sourcecode:: sh
	
	$ git clone https://code.google.com/p/arachnid/

Every time you update C or Cython code, you may run the following command (rather that reinstall)

.. sourcecode:: sh
	
	$ python setup.py build_ext --inplace
	
Every time you add a new script or the first time you get the source, you may run the following command (rather that reinstall)

.. sourcecode:: sh

	$ python setup.py develop --install-dir $BIN -m

.. _contribute:

Contributing
=============

Bug Report
----------

As a user or developer, you can report bugs or request features on the `Google Issues Tab <http://code.google.com/p/arachnid/issues/entry>`_. Please,
search for your issue before submitting it.

Source Code
-----------

The preferred way to contribute to Arachnid is to create your own local version or fork on code.google.com.

#. `Get a google account <https://accounts.google.com/NewAccount>`_
#. `Create a google code project <http://code.google.com/hosting/createProject>`_
#. Make a local copy of this project

	.. sourcecode:: sh
	
		$ git clone https://code.google.com/p/arachnid/

#. Work on your copy locally using Git for version control

	.. sourcecode:: sh
		
		# First time only, create a branch with label `my-feature`
		$ git remote add origin https://my-username@code.google.com/p/arachnid-my-fork-name/ 
		$ git push origin master
		$ git checkout -b my-feature 
		
		# Run for each new file
		$ git add modified_files
		
		# Run for each major change
		$ git commit
		
		# Run to push to public respository (or your respository)
		$ git push origin master

#. When your feature is ready for release, `create an issue <http://code.google.com/p/arachnid/issues/entry>`_
	
	- Use the `Review Request` template and include the URL to your project and a description and your contact.
	
	- You may be invited to act as a contributor or your code will be incorporated.

Your code should be error free and conform to the current code (also avoid `import *`). You can use the 
following tools to help ensure your code conforms to the
proper standards.

 - Basic documentation and error checking

	.. sourcecode:: sh
	
		$ easy_install pyflakes
		$ pyflakes path/to/module.py

 - Unit test coverage

	.. sourcecode:: sh
	
		$ easy_install nose coverage
		$ nosetests --with-coverage path/to/tests_for_package

Debugging C/C++-code
---------------------

Memory errors are the bain of every C/C++ developers existence. One important tool to finding these 
errors is `valgrind <http://valgrind.org/>`_.

#. Install valgrind

#. Download and modify the Python suppressions file: `valgrind-python.supp <http://svn.python.org/projects/python/trunk/Misc/valgrind-python.supp>`_
   See the directions in the `README.valgrind <http://svn.python.org/projects/python/trunk/Misc/README.valgrind>`_ for more information on modifying
   the suppressions.

#. Run valgrind over your code (choose a small example because there is a performance cot)

.. sourcecode:: sh

	$ valgrind -v --suppressions=valgrind-python.supp python my_test_script.py

Documentation Hack
==================

To get the documentation to build correctly, you need to edit `sphinx/ext/autosummary/generate.py` in your site-packages
directory. 

Change Line 143 from

.. sourcecode:: py

	for name in dir(obj):

to

.. sourcecode:: py

	for name in vars(obj):

A little background: The default autosummary code gets all inherited members of a class. This ensures only the current
members will be documented.


Create a Program Script
=======================

.. example batch program
.. example file processor program

Under construction

------
TODO
------

.. todolist::



