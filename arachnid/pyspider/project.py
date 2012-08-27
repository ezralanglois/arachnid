''' Create a full pySPIDER reconstruction project

This |spi| batch file generates a directory structure and set of scripts to
run a full reference-based |spi| reconstruction from scratch.

The directory structure and placement of the scripts is illustrated below:

.. container:: bottomnav, topic

    |    project-name
    |     \|
    |     --- local
    |         \|
    |         --- ctf
    |         \|
    |         --- coords
    |         \|
    |         --- reference.cfg
    |         \|
    |         --- defocus.cfg
    |         \|
    |         --- autopick.cfg
    |         \|
    |         --- crop.cfg
    |     \|
    |     --- cluster
    |         \|
    |         --- win
    |         \|
    |         --- data
    |         \|
    |         --- output
    |         \|
    |         --- refinement
    |         \|
    |         --- align.cfg
    |         \|
    |         --- refine.cfg
    
Tips
====

 #. It is recommended you set :option:`shared_scratch`, :option:`home_prefix`, and :option:`local_scratch` for MPI jobs
 
 #. Set `--is-ccd` to True if the micrographs were collected on CCD and have not been processed.

Examples
========

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    Create a project directory and scripts
    
    $ spi-project mic_*.tif -o ~/my-projects/project01 -r emd_1001.map -e ter -w 4 --apix 1.2 --voltage 300 --cs 2.26 --xmag 59000 --pixel-diameter 220
    
    Create a project directory and scripts for micrographs collected on CCD
    
    $ spi-project mic_*.tif -o ~/my-projects/project01 -r emd_1001.map -e ter -w 4 --apix 1.2 --voltage 300 --cs 2.26 --xmag 59000 --pixel-diameter 220 --is-ccd

Critical Options
================

.. program:: spi-project

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames containing micrographs.
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`

.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output directory with project name
    
.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Path to SPIDER params file

.. option:: -r <FILENAME>, --raw-reference <FILENAME>
    
    Raw reference volume

.. option:: -e <str>, --ext <str>
    
    Extension for SPIDER (three characters)

.. option:: --is-ccd <BOOL>

    Set true if the micrographs were collected on a CCD (and have not been processed)

.. option:: --apix <FLOAT>
    
    Pixel size, A (Default: 0)

.. option:: --voltage <FLOAT>
    
    Electron energy, KeV (Default: 0)

.. option:: --cs <FLOAT>
    
    Spherical aberration, mm (Default: 0)

.. option:: --xmag <FLOAT>
    
    Magnification (Default: 0)

.. option:: --pixel-diameter <INT>
    
    Actual size of particle, pixels (Default: 0)

Advanced Options
================

.. option:: --bin-factor <float>
    
    Number of times to decimate params file, and parameters: `window-size`, `x-dist, and `x-dist` and optionally the micrograph

.. option:: -m <CHOICE>, --mpi-mode=('Default', 'All Cluster', 'All single node')

    Setup scripts to run with their default setup or on the cluster or on a single node", default=0

.. option:: --mpi-command <str>
    
    Command used to invoked MPI (Default: nohup mpiexec -stdin none -n $nodes -machinefile machinefile)

.. option:: --refine-step <LIST>

    Value of each parameter specified in the same order as `refine-name` for each round of refinement, e.g. 15,10:0:6:1,8:0:4,1:3:1

.. option:: --max-ref-proj <INT>

    Maximum number of reference projections in memory (Default: 5000)

.. option:: --restart-file <FILENAME>
    
    Set the restart file backing up processed files

.. option:: --worker-count <INT>
    
    Set number of  workers to process files in parallel

.. option:: --shared-scratch <FILENAME>

    File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)

.. option:: --home-prefix <FILENAME>
    
    File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)

.. option:: --local-scratch <FILENAME>
    
    File directory on local node to copy files (optional but recommended for MPI jobs)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`

.. todo:: unzip emd_1001.map.gz
.. todo:: download ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-1001/map/emd_1001.map.gz

.. todo:: add support for microscope log file

.. Created on Aug 16, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.metadata import spider_params
from ..core.app import program
from ..app import autopick
from ..util import crop
import reference, defocus, align, refine
import os, sys, glob

def batch(files, output, mpi_mode, mpi_command=None, **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
    
    files : list
            List of input filenames
    output : str
             Output directory ending with name of project
    mpi_mode : int
               MPI mode for the header of each script
    mpi_command : str
                  MPI command to run in MPI scripts
    extra : dict
            Unused keyword arguments
    '''
    
    run_single_node=\
    '''
     nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
     exit 0
    '''
    run_multi_node=\
    '''
     %s %%prog -c $PWD/$0 --use-MPI < /dev/null > `basename $0 cfg`log &
     exit 0
    '''%(mpi_command)
    run_hybrid_node = run_single_node
    
    if mpi_mode == 1:
        run_hybrid_node = run_multi_node
    elif mpi_mode == 2:
        run_multi_node = run_single_node
    
    sn_path = os.path.join(output, 'local')
    mn_path = os.path.join(output, 'cluster')
    
    write_config(files, run_single_node, run_hybrid_node, run_multi_node, sn_path, mn_path, **extra)
    
def write_config(files, run_single_node, run_hybrid_node, run_multi_node, sn_path, mn_path, raw_reference, ext, is_ccd, **extra):
    ''' Write out a configuration file for each script in the reconstruction protocol
    
    :Parameters:
    
    files : list
            List of input filenames\
    run_single_node : str
                      Command to run single node scripts
    run_hybrid_node : str
                      Command to run hybrid single/MPI scripts 
    run_multi_node : str
                     Command to run MPI scripts
    sn_path : str
              Path to files used in single node scripts only
    mn_path : str
              Path to files used in both MPI and single node scripts
    raw_reference : str
                    Filenam for raw input reference
    is_ccd : bool
             True if micrographs were collected on a CCD
    ext : str
          Extension for SPIDER files
    extra : dict
            Unused keyword arguments
    '''
    
    param = dict(
        param_file = os.path.join(mn_path, 'data', 'params'+ext),
        reference = os.path.join(mn_path, 'data', 'reference'),
        defocus_file = os.path.join(mn_path, 'data', 'defocus'),
        coordinate_file = os.path.join(sn_path, 'coords', 'sndc_0000000'+ext),
        output_pow = os.path.join(sn_path, "pow", "pow_00000"),
        stacks = os.path.join(mn_path, 'win', 'win_0000000'+ext),
        alignment = os.path.join(mn_path, 'refinement', 'align_0000'),
    )
    create_directories(param.values())
    param.update(extra)
    param.update(invert=is_ccd)
    del param['input_files']
    
    if len(param['refine_step']) == 0: del param['refine_step']
    
    tmp = [os.path.commonprefix(files)+'*']
    if len(glob.glob(tmp[0])) == len(files): 
        del files[:]
        files.extend(tmp)
    
    spider_params.write(param['param_file'], **extra)
    program.write_config(reference, 
                         input_files=[raw_reference], 
                         output=param['reference'], 
                         description=run_single_node,
                         config_path = sn_path,
                         **param)
    
    program.write_config(defocus,   
                         input_files=files, 
                         output=param['defocus_file'], 
                         description=run_hybrid_node, 
                         config_path = sn_path,
                         supports_MPI=True,
                         **param)
    
    program.write_config(autopick,   
                         input_files=files, 
                         output=param['coordinate_file'],
                         description=run_hybrid_node, 
                         config_path = sn_path,
                         supports_MPI=True,
                         **param)
    
    program.write_config(crop,   
                         input_files=files,
                         output = param['stacks'],
                         description=run_hybrid_node, 
                         config_path = sn_path,
                         supports_MPI=True,
                         **param)
    
    program.write_config(align,   
                         input_files=param['stacks'],
                         output = param['alignment'],
                         description = run_multi_node, 
                         config_path = mn_path,
                         supports_MPI=True,
                         **param)
    
    program.write_config(refine,   
                         input_files=param['stacks'],
                         output = param['alignment'],
                         description = run_multi_node, 
                         config_path = mn_path,
                         supports_MPI=True,
                         **param)

def create_directories(files):
    ''' Create directories for a set of files
    
    :Parameters:
    
    files : list
            List of filenames
    '''
    
    for filename in files:
        if filename is None: continue
        filename = os.path.dirname(filename)
        if not os.path.exists(filename):
            try: os.makedirs(filename)
            except: raise ValueError, "Error creating directory %s"%filename

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import OptionGroup
    
    if pgroup is None: pgroup=parser
    
    parser.add_option("-i", input_files=[],     help="List of input filenames containing micrographs", required_file=True, gui=dict(filetype="file-list"))
    parser.add_option("-o", output="",          help="Output directory with project name", gui=dict(filetype="save"), required=True)
    parser.add_option("-r", raw_reference="",   help="Raw reference volume", gui=dict(filetype="open"), required=True)
    parser.add_option("-e", ext="",             help="Extension for SPIDER (three characters)", gui=dict(filetype="open"), required=True)
    parser.add_option("", is_ccd=False,         help="Set true if the micrographs were collected on a CCD (and have not been processed)")
    parser.add_option("", apix=0.0,             help="Pixel size, A")
    parser.add_option("", voltage=0.0,          help="Electron energy, KeV")
    parser.add_option("", cs=0.0,               help="Spherical aberration, mm")
    parser.add_option("", xmag=0.0,             help="Magnification")
    parser.add_option("", pixel_diameter=0,     help="Actual size of particle, pixels")
        
    # Additional options to change
    group = OptionGroup(parser, "Additional Parameters", "Optional parameters to set", group_order=0,  id=__name__)
    group.add_option("-m", mpi_mode=('Default', 'All Cluster', 'All single node'), help="Setup scripts to run with their default setup or on the cluster or on a single node", default=0)
    group.add_option("",    mpi_command="nohup mpiexec -stdin none -n $nodes -machinefile machinefile", help="Command used to invoked MPI")
    group.add_option("",    refine_step=[],         help="Value of each parameter specified in the same order as `refine-name` for each round of refinement, e.g. 15,10:0:6:1,8:0:4,1:3:1")
    group.add_option("",    max_ref_proj=5000,      help="Maximum number of reference projections in memory")
    # todo - max_ref_proj for alignment should depend on reference count, if less!
    group.add_option("",    restart_file="",        help="Set the restart file backing up processed files",     gui=dict(filetype="open"))
    group.add_option("",    worker_count=0,         help="Set number of  workers to process files in parallel",  gui=dict(maximum=sys.maxint, minimum=0))
    group.add_option("",    shared_scratch="",      help="File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)", gui=dict(filetype="open"))
    group.add_option("",    home_prefix="",         help="File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)", gui=dict(filetype="open"))
    group.add_option("",    local_scratch="",       help="File directory on local node to copy files (optional but recommended for MPI jobs)", gui=dict(filetype="open"))
    parser.add_option_group(group)
    
def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    #spider_params.check_options(options) # interactive
    if len(options.ext) != 3: raise OptionValueError, "SPIDER extension must be three characters"

    if options.apix == 0.0:
        raise OptionValueError, "No pixel size in angstroms specified (--apix), either specifiy it or an existing SPIDER params file"
    if options.voltage == 0.0:
        raise OptionValueError, "No electron energy in KeV specified (--voltage), either specifiy it or an existing SPIDER params file"
    if options.cs == 0.0:
        raise OptionValueError, "No spherical aberration in mm specified (--cs), either specifiy it or an existing SPIDER params file"
    if options.xmag == 0.0:
        raise OptionValueError, "No magnification specified (--xmag), either specifiy it or an existing SPIDER params file"
    if options.pixel_diameter == 0.0:
        raise OptionValueError, "No actual size of particle in pixels specified (--pixel_diameter), either specifiy it or an existing SPIDER params file"
    

def main():
    #Main entry point for this script
    from ..core.app.program import run_hybrid_program
    
    run_hybrid_program(__name__,
        description = '''Generate all the scripts and directories for a pySPIDER project 
                        
                        $ %prog micrograph_files* -o project-name -r raw-reference -e extension -p params -w 4 --apix 1.2 --voltage 300 --cs 2.26 --xmag 59000 --pixel-diameter 220
                      ''',
        supports_MPI=False,
        use_version = False,
        max_filename_len = 78,
    )
if __name__ == "__main__": main()


