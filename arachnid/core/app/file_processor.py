''' Framework for independent processing of files in serial, in parallel on a workstation or on a cluster using MPI

The file processor module provides basic functionality to any program that processes a
set of files independently.

Parameters
++++++++++

The bench script has the following inheritable parameters:

.. option:: -l <int>, --id-len <int>
    
    Set the expected length of the document file ID

.. option:: -b <str>, --restart-file <str>
    
    Set the restart file to keep track of which processes have finished. If this is empty, then a restart file is 
    written to .restart.$script_name You must set this to restart (otherwise it will overwrite the automatic restart file).

.. option:: -w <int>, --work-count <int>
    
     Set the number of micrographs to process in parallel (keep in mind memory and processor restrictions)

..todo:: 
    
    - add new version of check dependencies for update to align and refine
    - test and implement check_dependencies

.. Created on Oct 16, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..parallel import mpi_utility
from ..metadata import spider_utility
import os, logging, sys, numpy
from progress import progress

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def main(files, module, restart_file="", **extra):
    '''Main driver for a generic file processor
    
    .. sourcecode:: py
    
        >>> import os
        >>> os.system('more data.csv')
        id,select,peak
        1,1,0.5
        2,0,0.1
        3,0,0.6
        4,1,0.7
        5,0,0.2

        >>> from core.app.file_processor.file_processor import *
        >>> from arachnid.util import bench
        >>> main(['data.csv'], bench, metrics=['positive'], pcol=['peak'])
        peak
        ====
        
        +------+----------+
        |  id  | positive |
        +======+==========+
        | data |  5.000   |
        +------+----------+
        
        Average
        =======
        
        +--------+----------+
        | column | positive |
        +========+==========+
        |  peak  |  5.000   |
        +--------+----------+
    
    :Parameters:
    
    files : list
            List of filenames, tuple groups or lists of filenames
    module : module
             Main module containing entry points
    restart_file : string
                  Restart filename
    extra : dict
            Unused extra keyword arguments
    '''
    
    extra['restart_file']=restart_file # require restart_file=restart_file?
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
        #extra['input_files']=files
        _logger.debug("Test dependencies2: %d"%len(files))
        #if len(files) > 1: files = restart(restart_file, files)
    else: extra['finished']=None
    _logger.debug("Start processing1")
    tfiles = mpi_utility.broadcast(files, **extra)
    if not mpi_utility.is_root(**extra):
        tfiles = set([os.path.basename(f) for f in tfiles])
        files = [f for f in files if f in tfiles]
    _logger.debug("Start processing2")
        
    
    '''
    if mpi_utility.is_root(**extra):
        if restart_file == "": restart_file = ".restart.%s"%module.__name__
        try: 
            fout = file(restart_file, "a") if len(files) > 1 and restart_file != "" else None
        except: fout = None
        #if fout is not None:
        #    write_dependencies(fout, files, **extras)
    '''
    
    extra['finished'] = mpi_utility.broadcast(extra['finished'], **extra)
    if initialize is not None:
        #if mpi_utility.is_root(**extra):
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
            #_logger.critical("progress-report: %d,%d"%(current, len(files)))
            try:
                monitor.update()
                if reduce_all is not None:
                    current += 1
                    filename = reduce_all(filename, file_index=index, file_count=len(files), file_completed=current, **extra)
                    _logger.info("Finished: %d,%d - Time left: %s - %s"%(current, len(files), monitor.time_remaining(True), str(filename)))
                else:
                    _logger.info("Finished: %d,%d - Time left: %s"%(current, len(files), monitor.time_remaining(True)))
                '''
                if fout is not None:
                    if not isinstance(filename, list) and not isinstance(filename, tuple): filename = [filename]
                    #for f in filename: fout.write("%s:%d\n"%(str(f), os.path.getctime(f)))
                    for f in filename: fout.write("%s\n"%(str(f)))
                    fout.flush()
                '''
            except:
                _logger.exception("Error in root process")
                del files[:]
    if len(files) == 0:
        raise ValueError, "Error in root process"
    if mpi_utility.is_root(**extra):
        if finalize is not None: finalize(files, **extra)
    '''
    if mpi_utility.is_root(**extra) and fout is not None:
        fout.close()
        if restart_file != "" and os.path.exists(restart_file): os.unlink(restart_file)
    '''

def restart(filename, files):
    '''Test if script can restart and update file list appropriately
    
    .. sourcecode:: py

        >>> import os
        >>> os.system('more restart.txt')
        win_00001.spi
        
        >>> from core.app.file_processor.file_processor import *
        >>> restart('restart.txt', ['win_00001.spi','win_00002.spi','win_15644.spi','win_15646.spi'])
        ['win_00002.spi','win_15644.spi','win_15646.spi']
    
    Restart will not occur if:
    
    #. No restart file is given
    #. Most options are changed
    #. Input files have been modified
    #. Output files are missing
    
    :Parameters:
    
    filename : string
               Restart file
    files : list
            List of files to test for completion
    
    :Returns:

    val : list
          List of files left to processed
    '''
    
    if os.path.exists(filename):
        _logger.debug("Found restart file: %s"%filename)
        fin = open(filename, 'r')
        last = [line.strip() for line in fin]
        fin.close()
        
        last = set(last)
        if len(files) > 0 and isinstance(files[0], tuple):
            return [f for f in files if str(f[0]) not in last]
        return [f for f in files if f not in last]
    return files

def check_dependencies(files, infile_deps, outfile_deps, opt_changed, force=False, id_len=0, data_ext=None, restart_test=False, **extra):
    ''' Generate a subset of files required to process based on changes to input and existing
    output files.
    
    #. Check if output files exist
    #. Check modification times of output against input
    
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
    id_len : int
             Max length of SPIDER ID
    restart_test : bool
                   Test if program will restart
    extra : dict
            Unused extra keyword arguments
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
                if spider_utility.is_spider_filename(extra[out]) and spider_utility.is_spider_filename(f):
                    deps.append(spider_utility.spider_filename(extra[out], f, id_len))
                else: deps.append(extra[out])
        else:
            deps = [spider_utility.spider_filename(extra[out], f, id_len) for out in outfile_deps if out != "" and spider_utility.is_spider_filename(extra[out])]
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
        _logger.info("Skipping %d files - all dependencies satisfied (use --force or force: True to reprocess)"%len(finished))
    
    if restart_test:
        sys.exit(0)
    return unfinished, finished

def setup_options(parser, pgroup=None):
    # Options added to OptionParser by core.app.program
    from settings import OptionGroup
    group = OptionGroup(parser, "Processor", "Options to control the state of the file processor",  id=__name__)
    group.add_option("",   id_len=0,          help="Set the expected length of the document file ID",     gui=dict(maximum=sys.maxint, minimum=0))
    group.add_option("",   restart_file="",   help="Set the restart file backing up processed files",     gui=dict(filetype="open"), dependent=False)
    group.add_option("-w", worker_count=0,    help="Set number of  workers to process files in parallel",  gui=dict(maximum=sys.maxint, minimum=0), dependent=False)
    group.add_option("",   force=False,       help="Force the program to run from the start", dependent=False)
    group.add_option("",   restart_test=False,help="Test if the program will restart", dependent=False)
    pgroup.add_option_group(group)

def check_options(options):
    # Options tested by core.app.program before the program is run
    from settings import OptionValueError
    
    if len(options.input_files) < 1: raise OptionValueError, "Requires at least one file"
    
def supports(main_module):
    '''
    '''
    
    return hasattr(main_module, "process")

