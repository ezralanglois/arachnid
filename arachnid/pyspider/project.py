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
    
    # Create a project directory and scripts using 4 cores
    
    $ spi-project mic_*.tif -o ~/my-projects/project01 -r emd_1001.map -e ter -w 4 --apix 1.2 --voltage 300 --cs 2.26 --pixel-diameter 220
    
    # Create a project directory and scripts for micrographs collected on CCD using 4 cores
    
    $ spi-project mic_*.tif -o ~/my-projects/project01 -r emd_1001.map -e ter -w 4 --apix 1.2 --voltage 300 --cs 2.26 --pixel-diameter 220 --is-ccd

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

.. option:: --pixel-diameter <INT>
    
    Actual size of particle, pixels (Default: 0)

.. option:: --cs <FLOAT>
    
    Spherical aberration, mm (Default: 0)

.. option:: --scattering-doc <FILENAME>
    
    Filename for x-ray scatter file; set to ribosome for a default, 8A scattering file (optional, but recommended)

Advanced Options
================

.. option:: --xmag <FLOAT>
    
    Magnification, optional (Default: 0)

.. option:: --bin-factor <float>
    
    Number of times to decimate params file, and parameters: `window-size`, `x-dist, and `x-dist` and optionally the micrograph

.. option:: --worker-count <INT>
    
    Set number of  workers to process files in parallel

.. option:: -m <CHOICE>, --mpi-mode=('Default', 'All Cluster', 'All single node')

    Setup scripts to run with their default setup or on the cluster or on a single node", default=0

.. option:: --mpi-command <str>
    
    Command used to invoked MPI, if empty, then attempt to detect version of MPI and provide the command

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

.. todo:: max_ref_proj for alignment should depend on reference count, if less!

.. todo:: detect MPI command and set options based on whether is mpich or openmpi

.. Created on Aug 16, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.metadata import spider_params
from ..core.app import program
from ..app import autopick
from ..util import crop
import reference, defocus, align, refine
import os, glob, logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

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
    
    if mpi_command == "": mpi_command = detect_MPI()
    run_single_node=\
    '''
     %(prog)s -c $PWD/$0 > `basename $0 cfg`log
     exit 0
    '''
    run_multi_node=\
    '''
     %s %s -c $PWD/$0 --use-MPI < /dev/null > `basename $0 cfg`log
     exit 0
    '''%(mpi_command, '%(prog)s')
    run_hybrid_node = run_single_node
    
    if mpi_mode == 1:
        run_hybrid_node = run_multi_node
    elif mpi_mode == 2:
        run_multi_node = run_single_node
    
    sn_path = os.path.join(output, 'local')
    mn_path = os.path.join(output, 'cluster')
    
    write_config(files, run_single_node, run_hybrid_node, run_multi_node, sn_path, mn_path, output, **extra)
    
def write_config(files, run_single_node, run_hybrid_node, run_multi_node, sn_path, mn_path, output, raw_reference, ext, is_ccd, **extra):
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
    output : str
             Output directory root
    raw_reference : str
                    Filenam for raw input reference
    is_ccd : bool
             True if micrographs were collected on a CCD
    ext : str
          Extension for SPIDER files
    extra : dict
            Unused keyword arguments
    '''
    
    mn_base = os.path.basename(mn_path)
    sn_base = os.path.basename(sn_path)
    param = dict(
        param_file = os.path.join(mn_base, 'data', 'params'+ext),
        reference = os.path.join(mn_base, 'data', 'reference'),
        defocus_file = os.path.join(mn_base, 'data', 'defocus'),
        coordinate_file = os.path.join(sn_base, 'coords', 'sndc_0000000'+ext),
        output_pow = os.path.join(sn_base, "pow", "pow_00000"),
        stacks = os.path.join(mn_base, 'win', 'win_0000000'+ext),
        alignment = os.path.join(mn_base, 'refinement', 'align_0000'),
    )
    create_directories(param.values())
    param.update(extra)
    param.update(invert=is_ccd)
    del param['input_files']
    
    tmp = os.path.commonprefix(files)+'*'
    if len(glob.glob(tmp)) == len(files): files = [tmp]
    if extra['scattering_doc'] == "ribosome":
        extra['scattering_doc'] = download("http://www.wadsworth.org/spider_doc/spider/docs/techs/xray/scattering8.tst", os.path.join(mn_base, 'data'))
    elif extra['scattering_doc'] == "":
        _logger.warn("No scattering document file specified: `--scattering-doc`")
    
    spider_params.write(param['param_file'], **extra)
    
    modules = [(reference, dict(input_files=[raw_reference],
                               output=param['reference'],
                               description=run_single_node,
                               config_path = sn_path,
                               )), 
               (defocus,  dict(input_files=files,
                               output=param['defocus_file'],
                               description=run_hybrid_node, 
                               config_path = sn_path,
                               supports_MPI=True,
                               )),
                               
               (autopick, dict(input_files=files,
                               output=param['coordinate_file'],
                               description=run_hybrid_node, 
                               config_path = sn_path,
                               supports_MPI=True,
                               )), 
               (crop,     dict(input_files=files,
                               output = param['stacks'],
                               description=run_hybrid_node, 
                               config_path = sn_path,
                               supports_MPI=True,
                               )), 
               (align,    dict(input_files=param['stacks'],
                               output = param['alignment'],
                               description = run_multi_node, 
                               config_path = mn_path,
                               supports_MPI=True,
                               )), 
               (refine,    dict(input_files=param['stacks'],
                               output = param['alignment'],
                               description = run_multi_node, 
                               config_path = mn_path,
                               supports_MPI=True,
                               )),
                ]
    for mod, extra in modules:
        param.update(extra)
        program.write_config(mod, **param)
    
    module_type = {}
    for mod, extra in modules:
        type = extra['config_path']
        if type not in module_type: module_type[type]=[]
        module_type[type].append(mod)
    
    map = program.map_module_to_program()
    for path, modules in module_type.iteritems():
        type = os.path.basename(path)
        fout = open(os.path.join(output, 'run_%s'%type), 'w')
        fout.write("#!/bin/bash\n")
        for mod in modules:
            fout.write('sh %s\n'%os.path.join('..', path, map[mod.__name__]))
            fout.write('if [ "$?" != "0" ] ; then\nexit 1\nfi\n')
        fout.close()

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

def detect_MPI():
    ''' Detect if MPI is available, if not return None otherwise return command
    for OpenMPI or MPICH.
    
    :Returns:
    
    command : str
              Proper command for running Arachnid Scripts
    '''
    
    from subprocess import call
    
    ret = call('mpiexec --version', shell=True, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
    if ret == 0: # Detected OpenMPI
        return "mpiexec -stdin none -n $nodes -machinefile machinefile"
    return "mpiexec -n $nodes -machinefile machinefile"

def download(urlpath, filepath):
    '''Download the file at the given URL to the local filepath
    
    This function uses the urllib Python package to download file from to the remote URL
    to the local file path.
    
    :Parameters:
        
    urlpath : str
              Full URL to download the file from
    filepath : str
               Local path for filename
    
    :Returns:

    val : str
          Local filename
    '''
    import urllib
    from urlparse import urlparse
    
    filename = urllib.url2pathname(urlparse(urlpath)[2])
    filename = os.path.join(os.path.normpath(filepath), os.path.basename(filename))
    urllib.urlretrieve(urlpath, filename)
    return filename

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
    parser.add_option("", pixel_diameter=0,     help="Actual size of particle, pixels")
    parser.add_option("", cs=0.0,               help="Spherical aberration, mm")
    parser.add_option("", scattering_doc="",    help="Filename for x-ray scatter file; set to ribosome for a default, 8A scattering file (optional, but recommended)")
    parser.add_option("", xmag=0.0,             help="Magnification (optional)")
        
    # Additional options to change
    group = OptionGroup(parser, "Additional Parameters", "Optional parameters to set", group_order=0,  id=__name__)
    group.add_option("-m",  mpi_mode=('Default', 'All Cluster', 'All single node'), help="Setup scripts to run with their default setup or on the cluster or on a single node", default=0)
    group.add_option("",    mpi_command="",         help="Command used to invoked MPI, if empty, then attempt to detect version of MPI and provide the command")
    group.add_option("",    bin_factor=1.0,         help="Decimatation factor for the script: changes size of images, coordinates, parameters such as pixel_size or window unless otherwise specified")
    group.add_option("-w",  worker_count=0,         help="Set number of  workers to process files in parallel",  gui=dict(minimum=0))
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
    if options.pixel_diameter == 0.0:
        raise OptionValueError, "No actual size of particle in pixels specified (--pixel_diameter), either specifiy it or an existing SPIDER params file"
    

def main():
    #Main entry point for this script
    from ..core.app.program import run_hybrid_program
    
    run_hybrid_program(__name__,
        description = '''Generate all the scripts and directories for a pySPIDER project 
                        
                        $ %prog micrograph_files* -o project-name -r raw-reference -e extension -p params -w 4 --apix 1.2 --voltage 300 --cs 2.26 --pixel-diameter 220
                      ''',
        supports_MPI=False,
        use_version = False,
        max_filename_len = 78,
    )
if __name__ == "__main__": main()


