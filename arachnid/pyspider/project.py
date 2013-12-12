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
    |         \|
    |         --- project.cfg
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
 
 #. Set `--is-film` to True if the micrographs were collected on FILM or the CCD micrographs have been processed (inverted).

Examples
========

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Create a project directory and scripts using 4 cores
    
    $ spi-project mic_*.tif -o ~/my-projects/project01 -r emd_1001.map -e ter -w 4 --apix 1.2 --voltage 300 --cs 2.26 --pixel-diameter 220 --is-film
    
    # Create a project directory and scripts for micrographs collected on CCD using 4 cores
    
    $ spi-project mic_*.tif -o ~/my-projects/project01 -r emd_1001.map -e ter -w 4 --apix 1.2 --voltage 300 --cs 2.26 --pixel-diameter 220

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

.. option:: --is-film <BOOL>

    Set true if the micrographs were collected on a FILM (or have been processed)

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

.. option:: -e <str>, --ext <str>
    
    Extension for SPIDER, three characters (Default: dat)

.. option:: --xmag <FLOAT>
    
    Magnification, optional (Default: 0)

.. option:: --worker-count <INT>
    
    Set number of  workers to process files in parallel (Default: 0)

.. option:: -t <INT>, --thread-count <INT>
    
    Set number of threads to run in parallel, if not set then SPIDER uses all cores (Default: 0)

.. option:: -m <CHOICE>, --mpi-mode=('Default', 'All Cluster', 'All single node')

    Setup scripts to run with their default setup or on the cluster or on a single node

.. option:: --mpi-command <str>
    
    Command used to invoked MPI, if empty, then attempt to detect version of MPI and provide the command

.. option:: --shared-scratch <FILENAME>

    File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)

.. option:: --home-prefix <FILENAME>
    
    File directory accessible to all nodes to copy files, if empty then it uses the absolute path of the output file (optional but recommended for MPI jobs)

.. option:: --local-scratch <FILENAME>
    
    File directory on local node to copy files (optional but recommended for MPI jobs)
    
.. option:: --spider-path <FILENAME>

    Filename for SPIDER executable

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`

.. todo:: unzip emd_1001.map.gz
.. todo:: download ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-1001/map/emd_1001.map.gz

.. todo:: add support for microscope log file

.. todo:: where to find spider executable, add to installation discussion

.. Created on Aug 16, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.metadata import spider_params, spider_utility
from ..app import autopick
from ..util import crop, relion_selection #, mic_select
import reference, defocus, align, refine, legion_to_spider, autorefine, create_align
import os, logging, re #, glob

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

_cluster_modes=('pySPIDER', 'AutoRefine', 'Relion')
_cluster_default=2

def batch(files, output, leginon_filename="", leginon_offset=0, **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
    
    files : list
            List of input filenames
    output : str
             Output directory ending with name of project
    leginon_filename : str
                       Template for SPIDER softlink
    leginon_offset : int
                     Offset for SPIDER ID
    extra : dict
            Unused keyword arguments
    '''
    
    if legion_to_spider.is_legion_filename(files):
        data_ext = extra['ext']
        if data_ext[0]!='.': data_ext = '.'+data_ext
        leginon_filename = os.path.splitext(leginon_filename)[0]+data_ext
        old = files
        files = legion_to_spider.convert_to_spider(files, leginon_filename, leginon_offset)
        _logger.info("Added %d new files of %d"%(len(files), len(old)))
    
    
    _logger.info("Writing project to %s"%output)
    
    if extra['mpi_mode'] == 1:
        _logger.info("Creating multi-node project")
    elif extra['mpi_mode'] == 2:
        _logger.info("Creating single-node project")
    else:
        _logger.info("Creating hybrid project")

    
    if extra['worker_count'] > 1 and extra['thread_count'] == 0: extra['thread_count']=1
    del extra['input_files']
    
    id_len = spider_utility.spider_id_length(os.path.splitext(files[0])[0])
    if id_len == 0: raise ValueError, "Input file not a SPIDER file - id length 0"
    modules, param, config_path = workflow(files, output, id_len, **extra)
    
    write_config(modules, param, config_path, output, **extra)
    _logger.info("Completed")
    
def write_config(modules, param, config_path, output, support_project=True, **extra):
    ''' Write out a configuration file for each script in the reconstruction protocol
    
    :Parameters:
    
    files : list
            List of input filenames
    output : str
             Output directory root
    raw_reference : str
                    Filenam for raw input reference
    ext : str
          Extension for SPIDER files
    extra : dict
            Unused keyword arguments
    '''
    
    description_map = script_header(**extra)
    _logger.info("Creating directories")
    create_directories(output, param.values()+[os.path.join(os.path.basename(config_path['SN']), 'log', 'dummy'), os.path.join(os.path.basename(config_path['MN']), 'log', 'dummy')])
    
    _logger.debug("Writing SPIDER params file")
    spider_params.write_update(os.path.join(output, param['param_file']), **extra)

    map = {}
    for mod, extra in modules:
        param.update(extra)
        param.update(description=description_map[extra['run_type']])
        _logger.info("Writing config file for %s"%module_name(mod))
        map[mod.__name__] = program.update_config(mod, **param)
        if map[mod.__name__] == "":
            map[mod.__name__] = program.write_config(mod, **param)
    
    module_type = {}
    module_key=[]
    for mod, extra in modules:
        type = extra['config_path']
        
        if type not in module_type: 
            module_key.append(type)
            module_type[type]=[]
        module_type[type].append(mod)
    
    scripts = []
    #map = program.map_module_to_program()
    boutput = os.path.basename(output)
    for path in module_key:
        modules = module_type[path]
        type = os.path.basename(path)
        _logger.info("Writing script %s"%os.path.join(output, 'run_%s'%type))
        fout = open(os.path.join(output, 'run_%s'%type), 'w')
        progress_file = os.path.join(output, 'progress_%s.txt'%type)
        scripts.append((os.path.join(output, 'run_%s'%type), progress_file))
        fout.write("#!/bin/bash\n")
        if support_project:
            fout.write("sp-project -c local/project.cfg\n")
        if type == 'cluster':
            fout.write('MACHINEFILE="machinefile"\n')
            fout.write('if [ ! -e "$MACHINEFILE" ] ; then \n')
            fout.write('echo "Cannot find machinefile"\n')
            fout.write('exit 1\n')
            fout.write('fi\n')
            fout.write('nodes=`python -c "fin=open(\\"$MACHINEFILE\\", \'r\');lines=fin.readlines();print len([val for val in lines if val[0].strip() != \'\' and val[0].strip()[0] != \'#\'])"`\n')
            fout.write('export nodes MACHINEFILE\n')
            # count nodes
            # export environment variable
        fout.write('echo -n >%s\n'%(progress_file))
        for mod in modules:
            prog = os.path.join('..', map[mod.__name__]) if boutput != '.' and boutput != '' else map[mod.__name__]
            fout.write('sh %s\n'%prog)
            fout.write('if [ "$?" != "0" ] ; then\nexit 1\nfi\necho "%s">>%s\n'%(prog, progress_file))
        fout.close()
    return scripts

def workflow(input_files, output, id_len, raw_reference, ext='dat', cluster_mode=2, **extra):
    ''' Define the project workflow
    
    :Parameters:
    
    param : dict
            Parameter/value dictionary
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    modules : list
              List of tuples, modules and their specific parameters
    '''
    
    if 'particle_diameter' in extra and 'pixel_diameter' not in extra:
        extra['pixel_diameter'] = extra['particle_diameter']/extra['apix']
    
    data_ext = ext
    if ext[0] != '.': ext = '.'+ext
    else: data_ext = ext[1:]
    
    config_path=dict(
    SN = os.path.join(output, 'local'),
    MN = os.path.join(output, 'cluster'))
    setup_mpi_mode(config_path, **extra)
    
    mn_base = os.path.basename(config_path['MN'])
    sn_base = os.path.basename(config_path['SN'])
    param = dict(
        param_file = os.path.join(mn_base, 'data', 'params'+'.'+data_ext),
        reference = os.path.join(mn_base, 'data', 'reference'),
        defocus_file = os.path.join(mn_base, 'data', 'defocus'+'.'+data_ext),
        coordinate_file = os.path.join(sn_base, 'coords', 'sndc_'+"".zfill(id_len)+'.'+data_ext),
        output_pow = os.path.join(sn_base, "pow", "pow_"+"".zfill(id_len)+'.'+data_ext),
        output_mic = os.path.join(sn_base, "mic", "mic_dec_"+"".zfill(id_len)+'.'+data_ext),
        stacks = os.path.join(mn_base, 'win', 'win_'+"".zfill(id_len)+'.'+data_ext),
        alignment = os.path.join(mn_base, 'refinement', 'align_0000'),
        starfile = os.path.join(mn_base, 'data', 'data.star'),
    )
    
    
    stk = re.compile('0+')
    modules=[]
    if raw_reference != "":
        if spider_utility.is_spider_filename(raw_reference):
            param.update(reference=spider_utility.spider_filename(param['reference'], raw_reference))
        raw_reference = os.path.abspath(raw_reference)
        modules.extend([
                        (reference, dict(input_files=[raw_reference],
                               output=param['reference'],
                               run_type='SN',
                               ))
                        ])
    modules.extend( [
               (defocus,  dict(input_files=input_files,
                               output=param['defocus_file'],
                               run_type='HB',
                               supports_MPI=True,
                               )),
               (autopick, dict(input_files=input_files,
                               output=param['coordinate_file'],
                               run_type='HB',
                               supports_MPI=True,
                               )), 
               (crop,     dict(input_files=input_files,
                               output = param['stacks'],
                               run_type='HB',
                               supports_MPI=True,
                               )), 
                ])
    
    if cluster_mode == 0:
        modules.extend([
               (align,    dict(input_files=stk.sub('*', param['stacks']),
                               output = param['alignment'],
                               run_type='MN',
                               supports_MPI=True,
                               #select_file=format_utility.add_prefix(param['defocus_file'], "mic_select_")
                               )), 
               (refine,    dict(input_files=stk.sub('*', param['stacks']),
                               output = param['alignment'],
                               run_type='MN',
                               supports_MPI=True,
                               #selection_file
                               )),
        ])
    elif cluster_mode == 1:
        modules.extend([
               (create_align,  dict(input_files=stk.sub('*', param['stacks']),
                               output = param['alignment'],
                               run_type='SN',
                               supports_MPI=False,
                               #selection_file
                               )),
               (autorefine,    dict(input_files=stk.sub('*', param['stacks']),
                               output = param['alignment'],
                               run_type='MN',
                               supports_MPI=True,
                               #selection_file
                               )),
        ])
    else:
        #
        modules.extend([
               (relion_selection, dict(input_files=stk.sub('*', param['stacks']),
                               output = param['starfile'],
                               run_type='SN',
                               supports_MPI=False,
                               )),
        ])
        
    if 'home_prefix' in extra:
        if extra['home_prefix'] == "":
            extra['home_prefix'] = os.path.abspath(output)
        elif os.path.exists(os.path.join(extra['home_prefix'], output)):
            extra['home_prefix'] = os.path.join(extra['home_prefix'], output)
    
    extra['scattering_doc'] = copy_scattering_doc(config_path['MN'], **extra)
    
    for mod, settings in modules:
        # read config file
        prev=dict(settings)
        try: settings.update(program.read_config(mod, config_path=config_path[settings['run_type']]))
        except: _logger.exception("bug")
        settings.update(prev)
        settings.update(extra)
        settings.update(param)
        settings.update(invert=not extra['is_film'], data_ext=data_ext)
        settings.update(config_path=config_path[settings['run_type']])
        settings.update(log_file=os.path.join(os.path.basename(settings['config_path']), 'log', module_name(mod)+'.log'))
    
    return modules, param, config_path

def module_name(mod):
    ''' Return the name of the module
    
    :Parameters:
    
    mod : module
          Module to format
    
    :Returns:
    
    name : str
           Name of the module
    '''
    
    name = mod.__name__
    idx = name.rfind('.')
    if idx != -1: name = name[idx+1:]
    return name
    

def create_directories(output, files):
    ''' Create directories for a set of files
    
    :Parameters:
    
    output : str
             Root output directory
    files : list
            List of filenames
    '''
    
    for filename in files:
        if filename is None: continue
        filename = os.path.dirname(os.path.join(output, filename))
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
        return "mpiexec -stdin none -n $nodes -machinefile $MACHINEFILE"
    return "mpiexec -n $nodes -machinefile $MACHINEFILE"
    
def script_header(mpi_command="", **extra):
    '''
    
    :Parameters:
    
    mpi_command : str
                  MPI command to run in MPI scripts
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    header : str
             Header used to run the script
    '''
    
    if mpi_command == "": mpi_command = detect_MPI()
    description_map=dict( SN='''
     %(prog)s -c $PWD/$0 $@ > `basename $0 cfg`log
     exit $?
    ''',
    
    MN='''
     %s %s -c $PWD/$0 --use-MPI < /dev/null > .`basename $0 cfg`log
     exit $?
    '''%(mpi_command, '%(prog)s')
    )
    setup_mpi_mode(description_map, **extra)
    
    return description_map
        
def setup_mpi_mode(mode_map, mpi_mode=0, **extra):
    ''' Configure an MPI mode map
    
    :Parameters:
    
    mode_map : dict
               Dictionary to setup
    mpi_mode : int
               MPI mode for the header of each script
    extra : dict
            Unused keyword arguments
    '''
    
    mode_map.update(HB=mode_map['SN'])
    if mpi_mode == 1: mode_map.update(HB=mode_map['MN'])
    elif mpi_mode == 2: mode_map.update(MN=mode_map['SN'])

def copy_scattering_doc(output, scattering_doc="", ext="dat", **extra):
    ''' Download or copy a scattering document file to the proper location
    
    :Parameters:
    
    output : str
             Output dirctory for scattering doc
    scattering_doc : str, optional
                     Filename for scattering doc
    ext : str, optional
          Extension for SPIDER data files
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    scattering_doc : str
                     Filename for scattering doc
    '''
    
    if scattering_doc == "ribosome":
        scattering_doc = os.path.join(output, 'data', 'scattering8'+ext)
        if not os.path.exists(scattering_doc):
            _logger.info("Downloading scattering doc")
            tmp = download("http://www.wadsworth.org/spider_doc/spider/docs/techs/xray/scattering8.tst", os.path.join(output, 'data'))
            if ext != os.path.splitext(tmp)[1]:
                os.rename(tmp, scattering_doc)
        else:
            _logger.info("Downloading scattering doc - skipping, already found")
    elif scattering_doc == "":
        _logger.warn("No scattering document file specified: `--scattering-doc`")
    else:
        _logger.info("Copying scattering doc")
        fin = open(scattering_doc, 'r')
        scattering_doc = os.path.join(output, 'data', os.path.splitext(os.path.basename(scattering_doc))[0]+ext)
        fout = open(scattering_doc, 'w')
        for line in fin: fout.write(line)
        fin.close()
        fout.close
    return scattering_doc

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
        
    pgroup.add_option("-i", input_files=[],     help="List of input filenames containing micrographs", required_file=True, gui=dict(filetype="file-list"))
    pgroup.add_option("-o", output=".",         help="Output directory with project name", gui=dict(filetype="save"), required=True)
    pgroup.add_option("-r", raw_reference="",   help="Raw reference volume - optional", gui=dict(filetype="open"), required=False)
    pgroup.add_option("", is_film=False,        help="Set true if the micrographs were collected on film (or have been processed)", required=True)
    pgroup.add_option("", apix=0.0,             help="Pixel size, A", gui=dict(minimum=0.0, decimals=2, singleStep=0.1), required=True)
    pgroup.add_option("", voltage=0.0,          help="Electron energy, KeV", gui=dict(minimum=0), required=True)
    pgroup.add_option("", particle_diameter=0,  help="Longest diameter of the particle, angstroms", gui=dict(minimum=0), required=True)
    pgroup.add_option("", mask_diameter=0,      help="Longest diameter of the particle, angstroms", gui=dict(minimum=0), required=True)
    
    pgroup.add_option("", cs=0.0,               help="Spherical aberration, mm", gui=dict(minimum=0.0, decimals=2), required=True)
    
    
    # Additional options to change
    group = OptionGroup(parser, "Additional", "Optional parameters to set", group_order=0,  id=__name__)
    group.add_option("",    cluster_mode=_cluster_modes, help="Set the cluster mode", default=_cluster_default)
    group.add_option("",    scattering_doc="",      help="Filename for x-ray scatter file; set to ribosome for a default, 8A scattering file (optional, but recommended)", gui=dict(filetype="open"))
    group.add_option("",    window=0,               help="Set the window size: 0 means use 1.3*particle_diamater", gui=dict(minimum=0))
    group.add_option("",    curr_apix=0.0,          help="Current pixel size of the reference", gui=dict(minimum=0.0))
    group.add_option("",    xmag=0.0,               help="Magnification (optional)", gui=dict(minimum=0))
    group.add_option("-e",  ext="dat",              help="Extension for SPIDER (three characters)", required=True, gui=dict(maxLength=3))
    group.add_option("-m",  mpi_mode=('Default', 'All Cluster', 'All single node'), help="Setup scripts to run with their default setup or on the cluster or on a single node: ", default=0)
    group.add_option("",    mpi_command="",         help="Command used to invoked MPI, if empty, then attempt to detect version of MPI and provide the command")
    group.add_option("-w",  worker_count=0,         help="Set number of  workers to process files in parallel",  gui=dict(minimum=0))
    group.add_option("-t",  thread_count=0,         help="Set number of threads to run in parallel, if not set then SPIDER uses all cores",  gui=dict(minimum=0))
    group.add_option("",    shared_scratch="",      help="File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)", gui=dict(filetype="save"))
    group.add_option("",    home_prefix="",         help="File directory accessible to all nodes to copy files, if empty then it uses the absolute path of the output file (optional but recommended for MPI jobs)", gui=dict(filetype="open"))
    group.add_option("",    local_scratch="",       help="File directory on local node to copy files (optional but recommended for MPI jobs)", gui=dict(filetype="save"))
    group.add_option("",    local_temp="",          help="File directory on local node for temporary files (optional but recommended for MPI jobs)", gui=dict(filetype="save"))
    group.add_option("",    spider_path="",         help="Filename for SPIDER executable", gui=dict(filetype="open"))
    group.add_option("",    leginon_filename="mapped_micrographs/mic_0000000", help="Filename used to map legion files to SPIDER filenames")
    group.add_option("",    leginon_offset=0,       help="Offset for SPIDER id")
    parser.add_option_group(group)

def update_options(options):
    # Update the options
    
    setattr(options, 'noexit', True)
    options.create_cfg = os.path.join(options.output, 'local', 'project.cfg')
    
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
    if options.particle_diameter == 0.0:
        raise OptionValueError, "No longest diameter of the particle in angstroms specified (--particle-diameter), either specifiy it or an existing SPIDER params file"
    

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Generate all the scripts and directories for a pySPIDER project 
                        
                        $ %prog micrograph_files* -o project-name -r raw-reference -w 4 --apix 1.2 --voltage 300 --cs 2.26 --pixel-diameter 220
                      ''',
        supports_MPI=False,
        use_version = False,
        max_filename_len = 78,
    )
if __name__ == "__main__": main()

