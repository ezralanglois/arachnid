
===========
Development
===========

See :doc:`api` for code documentation.

Getting Source
==============

To get the latest version of the source, use the following `git <http://git-scm.com/documentation>`_ command:

.. sourcecode:: sh
	
	$ git clone https://github.com/ezralanglois/arachnid/arachnid.git

Alternatively, you can download a zip file of the latest development release:

.. raw:: html

	<a class="btn btn-primary" href="https://github.com/ezralanglois/arachnid/archive/master.zip" role="button">Download</a>


Every time you update C or Cython code, you may run the following command (rather that reinstall)

.. sourcecode:: sh
	
	$ python setup.py build_ext --inplace
	
Every time you add a new script or the first time you get the source, you may run the following command (rather that reinstall)

.. sourcecode:: sh

	$ python setup.py develop --install-dir $BIN -m

.. _contribute:

Reporting Bugs
==============

As a user or developer, you can report bugs or request features on the `Issues Tracker <https://github.com/ezralanglois/arachnid/issues>`_. Please,
search for your issue before submitting it.

Contributing Code
=================

The preferred way to contribute to Arachnid is to fork the main repository on Github and submit a "Pull Request".

#. `Create a GitHub account <https://github.com/signup/free>`_
#. `Fork the Arachnid repository <https://github.com/ezralanglois/arachnid/fork>`_
#. Clone a local copy to your machine
	
	.. sourcecode:: sh
		
		$ git clone git@github.com:<YourLogin>/arachnid/arachnid.git
	
	Replace <YourLogin> with your username for GitHub.

#. Create a branch to hold your changes

	.. sourcecode:: sh
		
		$ git checkout -b my-changes

#. When you have finished making changes, then

	.. sourcecode:: sh
		
		$ git add modified_file1 
		$ git add modified_file2
		$ git commit -m "Comment describing your modifications"

#. Upload your changes to your fork of Arachnid on GitHub

	.. sourcecode:: sh
	
		$ git push -u origin my-feature

#. Make a request to incorporate your changes 
	
	Go your fork of Arachnid on Github and click the "Pull request" button. This
	sends your changes to the maintainers for review.

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

Debugging C/C++ code
====================

Memory errors are the bain of every C/C++ developers existence. One important tool to finding these 
errors is `valgrind <http://valgrind.org/>`_.

#. Install valgrind

#. Download and modify the Python suppressions file: `valgrind-python.supp <http://svn.python.org/projects/python/trunk/Misc/valgrind-python.supp>`_
   See the directions in the `README.valgrind <http://svn.python.org/projects/python/trunk/Misc/README.valgrind>`_ for more information on modifying
   the suppressions.

#. Run valgrind over your code (choose a small example because there is a performance cot)

.. sourcecode:: sh

	$ valgrind -v --suppressions=valgrind-python.supp python my_test_script.py
	

Testing compilation of code
===========================

To test whether your code has any problems such as uninitalized variable, use the following flags in the environment for GCC type compilers.

.. sourcecode:: sh

	$ export CFLAGS="-Werror -Wno-unused-function -Wno-unknown-pragmas -Wno-format" CXXFLAGS="-Werror -Wno-format -Wno-unknown-pragmas -Wno-unused-function"
	$ export FFLAGS="-Werror -Wno-unused-function -Wtabs" F90FLAGS="-Werror -Wno-unused-function -Wtabs" F77FLAGS="-Werror -Wno-unused-function -Wtabs"

Packaging for Anaconda
======================

.. sourcecode:: sh
	
	$ conda skeleton pypi arachnid
	
	$ conda build arachnid/

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

------------
Custom Code
------------

Arachnid supports two types of scripting: applications and fast prototyping. Applications are scripts
that are designed for a user and to fit on the reconstruction workflow. Fast prototyping, however, is
a way to test new ideas, perform a simple task quickly or customize objects such as plots.

The :py:mod:`snippets` cover a set of examples for writing fast-prototype code. These
examples range from unstacking a SPIDER image stack to customizing the FSC plot.

------
TODO
------

.. todolist::



