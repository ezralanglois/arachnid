''' Framework for independent processing of files in serial, in parallel 
on a workstation or in parallel on a cluster using MPI.

The file processor module provides basic functionality to any program 
that processes a set of files independently. It can distribute processing
to multiple cores using the multiprocessing package as well as to multiple
nodes on a cluster using mpi2py.

Usage
-----

The target module must define the following functions:

.. py:function:: process(filename, **extra)

   Process a single file. During parallel processing this is invoked by a worker process.

   :param filename: Input filename to process
   :param extra: Unused keyword arguments (Options from the command line or config file, plus additional options)
   :returns: At minimum filename, however a tuple of additional arguments can be returned for processing by :py:func:`reduce_all`

The target module may include any of the following functions:

.. py:function:: initialize(files, extra)

   Initialize the input data before processing. During parallel processing this is invoked by all processes.

   :param files: List of input files to process
   :param extra: Dictionary of unused keyword arguments (Options from the command line or config file, plus additional options)
   :returns: None or a new list of files or objects to be processed by other routines
   
.. py:function:: finalize(files, **extra)

   Finalize data after processing. During parallel processing this is invoked by all processes.

   :param files: List of input files to process (possibly modified by :py:func:`initialize`)
   :param extra: Unused keyword arguments (Options from the command line or config file, plus additional options)

.. py:function:: reduce_all(filename, file_index=<int>, file_count=<int>, file_completed=<int>, **extra)

   Reduce data from worker to root process. During parallel processing this is invoked by the root process. Note that
   only filename is a positional argument.

   :param filename: Input filename that was processed. May be a tuple containing data resulting from :py:func:`process`.
   :param file_index: Current index in the list of input filenames
   :param file_count: Number of input filenames
   :param file_completed: Current count of finished files
   :param extra: Unused keyword arguments (Options from the command line or config file, plus additional options)
   :returns: Only the filename that was processed

.. py:function:: init_process(**extra)

    Initialize the input data before processing. During parallel processing this is invoked by all processes only
    once before processing except the root.

   :param extra: Keyword arguments (Options from the command line or config file, plus additional options)
   :returns: A dictionary of keywords to add or update.

.. py:function:: init_root(files, extra)

   Initialize the input data before processing (first init to be called). During parallel processing this is invoked only
   by the root process.

   :param files: List of input files to process
   :param extra: Dictionary of unused keyword arguments (Options from the command line or config file, plus additional options)
   :returns: None or a new list of files or objects to be processed by other routines

Each function also has access to the following keyword arguments:

    - finished: List of input files that have been processed and thus will not be processed this round
    - id_len: Maximum number of digits in the SPIDER ID

Parameters
----------

This module adds the following parameters to any script that
utilizes the program architecture.

.. option:: --id-len <int>
    
    Set the expected length of the document file ID

.. option:: -w <int>, --work-count <int>
    
     Set the number of micrographs to process in parallel (keep in mind memory and processor restrictions)

.. option:: --force <bool>
    
    Force the program to run from the start

.. option:: --restart-test <bool>
    
    Test if the program will restart

..todo:: 
    
    - add new version of check dependencies for update to align and refine

.. seealso:: 

    Module :py:mod:`arachnid.core.app.program`
        Main entry point for the Arachnid Program Architecture
    Module :py:mod:`arachnid.core.app.progress`
        Progress monitor for file processing
    Module :py:mod:`arachnid.core.app.settings`
        Program options parsing for command line and configuration file
    Module :py:mod:`arachnid.core.app.tracing`
        Logging controls
    Module :py:mod:`arachnid.core.parallel.mpi_utility`
        Handels parallizing both single node and multi-node processing

.. Created on Oct 16, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..parallel import mpi_utility
from ..metadata import spider_utility
import os, logging, sys, numpy
from progress import progress

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def main(files, module, **extra):
    '''Main driver for a generic file processor
    
    The workflow proceeds as follows.
    
    #. For the root process, call :py:func:`init_root` and update any input files
    #. Test if a subset of files have already been processed and remove these
    #. Broadcast new set of files to be processed to all nodes (if necessary)
    #. Broadcast files already processed to all nodes (if necessary)
    #. ??
    #. Initialize data for each process including root
    #. If no files, finalize and return
    #. Initialize data for each worker process
    #. Process files
    #. Reduce data to root worker
    #. Finalize data on root worker
    
    .. sourcecode:: py

        >>> from core.app import file_processor
        >>> from arachnid.util import crop
        >>> file_processor.main(['stack_01.spi'], crop)
    
    :Parameters:
    
    files : list
            List of filenames, tuple groups or lists of filenames
    module : module
             Main module containing entry points
    extra : dict
            Unused extra keyword arguments
    '''
    
    _logger.debug("File processer - begin")
    process, initialize, finalize, reduce_all, init_process, init_root = getattr(module, "process"), getattr(module, "initialize", None), getattr(module, "finalize", None), getattr(module, "reduce_all", None), getattr(module, "init_process", None), getattr(module, "init_root", None)
    monitor=None
    if mpi_utility.is_root(**extra):
        if init_root is not None:
            _logger.debug("Init-root")
            f = init_root(files, extra)
            if f is not None: files = f
        _logger.debug("Test dependencies1: %d"%len(files))
        files, finished = check_dependencies(files, **extra)
        extra['finished'] = finished
        _logger.debug("Test dependencies2: %d"%len(files))
    else: extra['finished']=None
    _logger.debug("Start processing1")
    tfiles = mpi_utility.broadcast(files, **extra)
    
    # Why?
    if not mpi_utility.is_root(**extra):
        tfiles = set([os.path.basename(f) for f in tfiles])
        files = [f for f in files if f in tfiles]
    _logger.debug("Start processing2")
    
    extra['finished'] = mpi_utility.broadcast(extra['finished'], **extra)
    if initialize is not None:
        _logger.debug("Init")
        f = initialize(files, extra)
        _logger.debug("Init-2")
        if f is not None: files = f
        #files = mpi_utility.broadcast(files, **extra)
    _logger.debug("Start processing3")
    if len(files) == 0:
        if mpi_utility.is_root(**extra):
            _logger.debug("No files to process")
            if finalize is not None: finalize(files, **extra)
        return
    
    if mpi_utility.is_root(**extra):
        _logger.debug("Setup progress monitor")
        monitor = progress(len(files))
         
    current = 0
    _logger.debug("Start processing")
    for index, filename in mpi_utility.mpi_reduce(process, files, init_process=init_process, **extra):
        if mpi_utility.is_root(**extra):
            try:
                monitor.update()
                if reduce_all is not None:
                    current += 1
                    filename = reduce_all(filename, file_index=index, file_count=len(files), file_completed=current, **extra)
                    _logger.info("Finished: %d,%d - Time left: %s - %s"%(current, len(files), monitor.time_remaining(True), str(filename)))
                else:
                    _logger.info("Finished: %d,%d - Time left: %s"%(current, len(files), monitor.time_remaining(True)))
            except:
                _logger.exception("Error in root process")
                del files[:]
    if len(files) == 0:
        raise ValueError, "Error in root process"
    if mpi_utility.is_root(**extra):
        if finalize is not None: finalize(files, **extra)

def check_dependencies(files, infile_deps, outfile_deps, opt_changed, force=False, id_len=0, data_ext=None, restart_test=False, **extra):
    ''' Generate a subset of files required to process based on changes to input and existing
    output files. Note that this dependency checking is similar to the program `make`.
    
    #. Check if output files exist
    #. Check modification times of output against input
    #. Check if `opt_changed` flag was set to True
    #. Check if `force` flag was set to True
    
    :Parameters:
    
    files : list
            List of input files
    infile_deps : list
                  List of input file dependencies
    outfile_deps : list
                   List of output file dependencies
    opt_changed : bool
                  If true, then options have changed; restart from beginning
    force : bool
            Force the program to restart from the beginning
    data_ext : str
               If the dependent file does not have an extension, add this extension
    id_len : int
             Max length of SPIDER ID
    restart_test : bool
                   Test if program will restart
    extra : dict
            Unused extra keyword arguments
            
    :Returns:
    
    unfinished : list
                 List of input filenames to process 
    finished : list
               List of input filenames that satisfy requirements and will not be processed.
    '''
    
    if opt_changed or force:
        msg = "configuration file changed" if opt_changed else "--force option specified"
        _logger.info("Skipping 0 files - restarting from the beginning - %s"%msg)
        if restart_test:
            sys.exit(0)
        return files, []
    unfinished = []
    finished = []
    if data_ext is not None and data_ext=="" and len(files) > 0:
        data_ext = os.path.splitext(files[0])[1]
        if len(data_ext) > 0: data_ext=data_ext[1:]
    for f in files:
        filename=f
        if isinstance(f, tuple): 
            f = f[0]
            try: f = int(f)
            except: pass
        if len(files) == 1:
            deps = []
            for out in outfile_deps:
                if out == "": continue
                if (spider_utility.is_spider_filename(extra[out]) or os.path.exists(spider_utility.spider_filename(extra[out], f, id_len))) and spider_utility.is_spider_filename(f):
                    deps.append(spider_utility.spider_filename(extra[out], f, id_len))
                else: deps.append(extra[out])
        else:
            deps = [spider_utility.spider_filename(extra[out], f, id_len) for out in outfile_deps if out != "" and (spider_utility.is_spider_filename(extra[out]) or os.path.exists(spider_utility.spider_filename(extra[out], f, id_len)))]
        if data_ext is not None:
            for i in xrange(len(deps)):
                if os.path.splitext(deps[i])[1] == "": deps[i] += '.'+data_ext
        exists = [os.path.exists(out) for out in deps]
        if not numpy.alltrue(exists):
            _logger.debug("Adding: %s because %s does not exist"%(f, deps[numpy.argmin(exists)]))
            unfinished.append(filename)
            continue
        mods = [os.path.getctime(out) for out in deps]
        if len(mods) == 0:
            _logger.debug("Adding: %s because no dependencies exist"%(f))
            unfinished.append(filename)
            continue
        first_output = numpy.min( mods )
        if len(files) == 1:
            
            deps = [f] if not isinstance(f, int) else []
            for input in infile_deps:
                if input == "": continue
                if spider_utility.is_spider_filename(extra[input]) and spider_utility.is_spider_filename(f):
                    deps.append(spider_utility.spider_filename(extra[input], f, id_len))
                else: 
                    deps.append(extra[input])
        else:
            deps = [f] if not isinstance(f, int) else []
            deps.extend([spider_utility.spider_filename(extra[input], f, id_len) for input in infile_deps if input != "" and spider_utility.is_spider_filename(extra[input])])
        if data_ext is not None:
            for i in xrange(len(deps)):
                if os.path.splitext(deps[i])[1] == "": deps[i] += '.'+data_ext
        mods = [os.path.getctime(input) for input in deps if input != "" and os.path.exists(input)]
        last_input = numpy.max( mods ) if len(mods) > 0 else 0
        if last_input >= first_output:
            _logger.debug("Adding: %s because %s has been modified in the future"%(f, deps[numpy.argmax(mods)]))
            unfinished.append(filename)
            continue
        else: finished.append(filename)
    if len(finished) > 0:
        #_logger.info("Skipping: %s all dependencies satisfied (use --force or force: True to reprocess)"%f)
        _logger.info("Skipping %d files - all dependencies satisfied (use --force or force: True to reprocess) - processing %d files"%(len(finished), len(unfinished)))
    
    if restart_test:
        sys.exit(0)
    return unfinished, finished

def setup_options(parser, pgroup=None):
    # Options added to OptionParser by core.app.program
    from settings import OptionGroup
    group = OptionGroup(parser, "Processor", "Options to control the state of the file processor",  id=__name__)
    group.add_option("",   id_len=0,          help="Set the expected length of the document file ID",     gui=dict(maximum=sys.maxint, minimum=0))
    group.add_option("-w", worker_count=0,    help="Set number of  workers to process files in parallel",  gui=dict(maximum=sys.maxint, minimum=0), dependent=False)
    group.add_option("",   force=False,       help="Force the program to run from the start", dependent=False)
    group.add_option("",   restart_test=False,help="Test if the program will restart", dependent=False)
    pgroup.add_option_group(group)

def check_options(options):
    # Options tested by core.app.program before the program is run
    from settings import OptionValueError
    
    if len(options.input_files) < 1: raise OptionValueError, "Requires at least one file"
    
def supports(main_module):
    ''' Test if module has the minimum required entry points
    to support file processing.
    
    :Parameters:
    
    main_module : module
                  Module containing entry points
    
    :Returns:
    
    flag : bool
           True if module has `process` function
           
    '''
    
    return hasattr(main_module, "process")

