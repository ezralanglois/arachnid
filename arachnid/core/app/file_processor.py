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

..todo:: add file modification time to restart file, use spider id to check for file

.. Created on Oct 16, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..parallel import mpi_utility
import os, logging, sys

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
    process, initialize, finalize, reduce_all = getattr(module, "process"), getattr(module, "initialize", None), getattr(module, "finalize", None), getattr(module, "reduce_all", None)
    if len(files) > 1: files = restart(restart_file, files)
    if initialize is not None: 
        f = initialize(files, extra)
        if f is not None: files = f
    
    if mpi_utility.is_root(**extra):
        if restart_file == "": restart_file = ".restart.%s"%module.__name__
        try: 
            fout = file(restart_file, "a") if len(files) > 1 and restart_file != "" else None
        except: fout = None
    
    current = 0
    for index, filename in mpi_utility.mpi_reduce(process, files, **extra):
        if mpi_utility.is_root(**extra):
            if reduce_all is not None: 
                current += 1
                filename = reduce_all(filename, file_index=index, file_count=len(files), file_completed=current, **extra)
            if fout is not None:
                if not isinstance(filename, list) and not isinstance(filename, tuple): filename = [filename]
                for f in filename: fout.write(str(f)+"\n")
                fout.flush()
    
    if mpi_utility.is_root(**extra):
        if finalize is not None: finalize(files, **extra)
    if mpi_utility.is_root(**extra) and fout is not None:
        fout.close()
        if restart_file != "" and os.path.exists(restart_file): os.unlink(restart_file)

def restart(filename, files):
    '''Test if script can restart and update file list appropriately
    
    .. sourcecode:: py

        >>> import os
        >>> os.system('more restart.txt')
        win_00001.spi
        
        >>> from core.app.file_processor.file_processor import *
        >>> restart('restart.txt', ['win_00001.spi','win_00002.spi','win_15644.spi','win_15646.spi'])
        ['win_00002.spi','win_15644.spi','win_15646.spi']
    
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
        fin = open(filename, 'r')
        last = set([line.strip() for line in fin])
        fin.close()
        if len(files) > 0 and isinstance(files[0], tuple):
            return [f for f in files if str(f[0]) not in last]
        return [f for f in files if f not in last]
    return files

def setup_options(parser, pgroup=None):
    # Options added to OptionParser by core.app.program
    from settings import OptionGroup
    group = OptionGroup(parser, "File Processor", "Options to control the state of the file processor",  id=__name__) if pgroup is None else pgroup
    group.add_option("",   id_len=0,          help="Set the expected length of the document file ID",     gui=dict(maximum=sys.maxint, minimum=0))
    group.add_option("",   restart_file="",   help="Set the restart file backing up processed files",     gui=dict(filetype="open"))
    group.add_option("-w", worker_count=0,    help="Set number of  workers to process files in parallel",  gui=dict(maximum=sys.maxint, minimum=0))
    if pgroup is None: parser.add_option_group(group)

def check_options(options):
    # Options tested by core.app.program before the program is run
    from settings import OptionValueError
    
    if len(options.input_files) < 1: raise OptionValueError, "Requires at least one file"
    
def supports(main_module):
    '''
    '''
    
    return hasattr(main_module, "process")

