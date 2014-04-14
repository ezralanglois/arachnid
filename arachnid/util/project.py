''' Create a project workflow

This script generates a workflow to process data collected by single-particle
cryo-electron microscopy. It generates a set of configuration files that can
be used to tune individual parameters as well as a directory structure
to organize all data.

A project consists of four primary directories

#. cfg - contains all configuration file for the scripts
#. linked - contains renamed links
#. cluster - contains all files to copy to the cluster
#. local - contains all files that can be copied to a local machine (e.g. laptop)

The directory structure and location of configuration files and scripts is
illustrated below.

.. container:: bottomnav, topic

    |    cfg/
    |     - project.cfg
    |     - enumerate_files.cfg
    |     - reference.cfg
    |     - align_frames.cfg
    |     - estimate_ctf.cfg
    |     - autopick.cfg
    |     - crop.cfg
    |     - crop_frame.cfg
    |     - autorefine.cfg
    |     - vicer.cfg
    |     - relion_selection.cfg
    |
    |    other/
    |
    |     
    |    cluster/
    |     - data/
    |         - data.star
    |         - reference.dat
    |     - win/
    |         - win_00001.dat
    |         - win_00002.dat
    |
    |    local/
    |     - coords/
    |        - sndc_00001.dat
    |        - sndc_00002.dat
    |
    |     - ctf/
    |        - pow/
    |           - pow_000001.dat
    |           - pow_000002.dat
    |
    |        - params.dat
    |        - ctf.dat
    |
    |      - mics_small/
    |      - mics/
    |
    |    run.sh


The workflow runs the following scripts in order:

    #. project
    #. enumerate_files  (optional)
    #. reference (optional)
    #. micrograph_alignment  (optional)
    #. Defocus
    #. AutoPick
    #. particle_alignment    (optional)
    #. Crop
    #. Refine
    #. ViCer
    #. selrelion
    #. Crop_frames

Single master config file - link input/output
Individual files - configure each program

Development
-----------

See :ref:`add-to-workflow`.

.. Created on Oct 20, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.image import ndimage_file
from ..core.metadata import spider_params
import logging
import os
import sys
from ..core.spider import spider

_project = sys.modules[__name__]

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, is_film=False, **extra):
    ''' Main entry point for a batch script.
    
    This function builds the workflow, writes configuration files
    and writes the workflow script.
    
    :Parameters:
    
    files : list
            List of input exposure images (or movie-mode stacks of frames)
    is_film : bool
              Disable contrast inversion
    extra : dict
            Unused keyword arguments
    '''
    
    extra['invert'] = not is_film
    workflow = build_workflow(files, extra)
    _logger.info("Work flow includes %d steps"%len(workflow))
    write_config(workflow, **extra)
    write_workflow(workflow)
    write_relion_settings(**extra)
    # run option
    
def write_relion_settings(align_file, mask_diameter, apix, **extra):
    ''' Generate a relion settings file
    
    :Parameters:
    
        align_file : str
                     Filename for output relion star file
        mask_diameter : float
                        Diameter of mask in angstroms
        apix : float
               Pixel size in angstroms
    extra : dict
            Keyword arguments
    '''
    
    fout = open('.lastrun_gui2d.settings', 'w')
    fout.write("Input images STAR file: == %s\n"%align_file)
    fout.write("Particle mask diameter (A): == %s\n"%mask_diameter)
    fout.write("Pixel size (A): == %s\n"%apix)
    fout.close()

def workflow_settings(files, param):
    ''' Generate a list of program objects for each script
    in the workflow.
    
    :Parameters:
    
    files : list
            List of input exposure images (or movie-mode stacks of frames)
    param : dict
            Parameter value pairs already set in the GUI
    
    :Returns:
    
    mods : list
           List of modules
    '''
    
    extra = vars(default_settings())
    param['input_files_compressed']=program.settings.compress_filenames(param['input_files'])
    param['input_files']=extra['input_files'].__class__(param['input_files'])
    extra.update(param)
    workflow = build_workflow(files, extra)
    param.update(extra)
    
    prog = program.generate_settings_tree(workflow[0][0], **extra)
    
    prog.update(param)
    param.update(vars(prog.values))
    spider_params.write_update(param['param_file'], **param)
    mods=[prog]
    first_param, first_script = find_root(workflow, ('unenum_files', 'movie_files', 'micrograph_files'))
    param[first_param] = param['input_files']
    prog.update({first_param: param['input_files']})
    for mod in workflow[1:]:
        prog = program.generate_settings_tree(mod[0], **param)
        assert(prog is not None)
        mods.append(prog)
        if len(param['input_files']) > 0 and mod[0] == first_script[0]:
            param['input_files'] = []
    return mods

def default_settings():
    ''' Generate a settings object from the default
    project settings.
    
    :Returns:
    
    values : object
             Settings object
    '''
    
    return program.generate_settings_tree(_project).values

def write_workflow(scripts):
    ''' Write out scripts necessary to run the workflow
    
    :Parameters:
    
    scripts : list
              List of scripts to run
    '''
    
    fout = open('run.sh', 'w')
    try:
        fout.write("#!/bin/bash\n")
        for script in scripts[1:]:
            if hasattr(script, 'configFile'):
                fout.write('sh %s\n'%script.configFile())
            else:
                fout.write('sh %s\n'%script[1])
            fout.write('if [ "$?" != "0" ] ; then\nexit $?\nfi\n')
    finally:
        fout.close()

def build_dependency_tree(scripts, root, found=[]):
    ''' Build a dependency tree to order all the scripts based on
    the name of input/output filenames.
    
    :Parameters:
    
    scripts : list
              List of tuples containing: module, script name, 
              input file deps, output file deps.
    root : tuple
           Root script whose output defines the next layers input. This
           tuple contains module,script-name,indeps,outdeps
    found : list
            List of previously found dependencies
    
    :Returns:
    
    scripts : list
              Ordered name of scripts
    '''
    
    branch=[]
    outdeps = set(root[3]+found)
    found=[]
    for script in scripts:
        intersect=outdeps&set(script[2])
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug("%s -> %s | %d == %d | %s == %s"%(root[1], script[1], len(outdeps), len(script[2]), str(outdeps), str(script[2])))
        if len(intersect)==len(script[2]):
            branch.append(script)
        else:
            found.extend(intersect)
    for b in branch:
        del scripts[scripts.index(b)]
    if len(scripts) > 0:
        for script in branch:
            current=list(found)
            if len(scripts) == 0: break
            branch.extend(build_dependency_tree(scripts, script, current))
    return branch

def build_workflow(files, extra):
    ''' Collect all modules that belong to the current workflow
    
    This modules searchs specific packages that contain application
    scripts and includes those that meet the following criterion:
        
        #. Must have a :py:func:`supports` function
        #. And this function must return true for the given parameters
    
    :Parameters:
    
    files : list
            List of input image files
    extra : dict
            Option, value pairs - can be updated
    
    :Returns:
    
    workflow : list
               List of programs
    '''
    
    import pkgutil
    from .. import app, pyspider, util
    
    workflow = [_project]
    for pkg in [app, pyspider, util]:
        modules = [name for _, name, ispkg in pkgutil.iter_modules([os.path.dirname(cpkg.__file__) for cpkg in [pkg]] ) if not ispkg]
        for name in modules:
            try:
                mod = getattr(__import__(pkg.__name__, globals(), locals(), [name]), name)
            except:pass
            else:
                if hasattr(mod, 'supports') and (not callable(mod.supports) or mod.supports(files, **extra)):
                    workflow.append(mod)
    for i in xrange(len(workflow)):
        workflow[i] = [workflow[i]]+list(program.collect_file_dependents(workflow[i], **extra))
    
    first_param,first_script = find_root(workflow, ('unenum_files', 'movie_files', 'micrograph_files'))
    # Hack
    if first_param == 'unenum_files':
        input2 = find_root(workflow, ('movie_files', 'micrograph_files'))[0]
        extra['linked_files'] = os.path.join('other', *os_path_split(extra[input2])[1:])
        extra[input2] = extra['linked_files']
        input2 = find_root(workflow, ('movie_files', 'micrograph_files'))[0]
        first_script[3] = ['--'+input2.replace('_', '-'), ]+first_script[3]
    
    return [workflow[0]]+build_dependency_tree(workflow[1:], workflow[0], ['--'+first_param.replace('_', '-')])

def os_path_split(path):
    ''' Split a filename into a list of directories (possibly) ending with a filename
    
    :Parameters:
        
        path : str
               Path to split
    
    :Returns:
        
        paths : list
                List of directories (possibly) ending with a filename
    
    .. note::
        
        Adopted from:
        http://stackoverflow.com/questions/4579908/cross-platform-splitting-of-path-in-python
    '''
    
    parts = []
    while True:
        newpath, tail = os.path.split(path)
        if newpath == path:
            assert not tail
            if path: parts.append(path)
            break
        parts.append(tail)
        path = newpath
    parts.reverse()
    return parts

def find_root(scripts, root_inputs):
    ''' Find the input for the root script
    
    :Parameters:
    
    scripts : list
              List of tuples containing: module, script name, 
              input file deps, output file deps.
    root_inputs : tuple
                  Possible roots in order of precendence
    
    :Returns:
    
    root_input : str
                 Root input
    '''
    
    for root_input in root_inputs:
        root_input2 = '--'+root_input.replace('_', '-')
        for scrpt in scripts:
            if root_input2 in scrpt[2]: return root_input, scrpt
    raise ValueError, "No root found!"
    
def write_config(workflow, **extra):
    ''' Write configurations files for each script in the
    workflow including the SPIDER CTF params file
    
    :Parameters:
    
    workflow : list
               List of modules
    extra : dict
            Keyword arguments
    '''
    
    spider_params.write_update(extra['param_file'], **extra)
    config_path=extra['config_path']
    if not os.path.exists(config_path):
        try: os.makedirs(config_path)
        except: pass
    
    first_param, first_script = find_root(workflow, ('unenum_files', 'movie_files', 'micrograph_files'))
    if first_param in extra: del extra[first_param]
    for mod, _, _, _ in workflow:
        param = program.update_config(mod, **extra)
        if param is not None:
            param.update(extra)
        else: param=extra
        program.write_config(mod, **param)
        if len(param['input_files']) > 0 and mod == first_script[0]:
            param['input_files'] = []
            
def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import OptionGroup
        
    pgroup.add_option("-i", input_files=[],      help="List of input filenames containing raw micrographs or stacks of micrograph frames", required_file=True, gui=dict(filetype="open"), dependent=False)
    pgroup.add_option("-r", raw_reference_file="", help="Raw reference volume - optional", gui=dict(filetype="open"), required=False)
    pgroup.add_option("-g", gain_file="",        help="Gain reference image for movie mode (must be a normalization image!) - optional", gui=dict(filetype="open"), required=False)
    pgroup.add_option("", is_film=False,         help="Set true if the micrographs have already been contrast inverted, usually when collected on film")
    pgroup.add_option("", apix=0.0,              help="Pixel size, Angstroms", gui=dict(minimum=0.0, decimals=4, singleStep=0.1), required=True)
    pgroup.add_option("", voltage=0.0,           help="Electron energy, KeV", gui=dict(minimum=0.0, singleStep=1.0), required=True)
    pgroup.add_option("", cs=0.0,                help="Spherical aberration, mm", gui=dict(minimum=0.0, decimals=2), required=True)
    pgroup.add_option("", window=0,              help="Set the window size (pixels): 0 means use 1.3*particle_diamater", gui=dict(minimum=0))
    pgroup.add_option("", particle_diameter=0.0, help="Longest diameter of the particle, Angstroms", gui=dict(minimum=0), required=True)
    pgroup.add_option("", mask_diameter=0.0,     help="Set the mask diameter (Angstroms): 0 means use 1.1*particle_diamater", gui=dict(minimum=0))
    
    addgroup = OptionGroup(parser, "Parallel", "Options for parallel processing")
    addgroup.add_option("-w", worker_count=1,  help="Set number of  workers to process files in parallel",  gui=dict(minimum=0), dependent=False)
    addgroup.add_option("-t", thread_count=1,  help="Number of threads per machine, 0 means determine from environment", gui=dict(minimum=0), dependent=False)
    pgroup.add_option_group(addgroup)
    
    shrgroup = OptionGroup(parser, "Metadata", "Files created during workflow")
    shrgroup.add_option("", config_path="cfg",                                     help="Location for configuration scripts", gui=dict(filetype="open"))
    shrgroup.add_option("", linked_files="",                                       help="Location for renamed links - name automatically set based on input", gui=dict(filetype="open"))
    shrgroup.add_option("", micrograph_files="other/mics/mic_00000.dat",          help="Location for micrograph files", gui=dict(filetype="open"))
    shrgroup.add_option("", param_file="local/ctf/params.dat",                     help="Location for SPIDER params file", gui=dict(filetype="save"))
    shrgroup.add_option("", movie_files="local/frames/mic_00000.dat",              help="Location for micrograph frame stacks", gui=dict(filetype="open"))
    shrgroup.add_option("", coordinate_file="local/coords/sndc_000000.dat",        help="Location for particle coordinates on micrograph", gui=dict(filetype="open"))
    shrgroup.add_option("", particle_file="cluster/win/win_000000.dat",            help="Location for windowed particle stacks", gui=dict(filetype="open"))
    shrgroup.add_option("", ctf_file="local/ctf/ctf.dat",                          help="Location of estimated CTF parameters per micrograph", gui=dict(filetype="open"))
    shrgroup.add_option("", pow_file="local/ctf/pow/pow_000000.dat",               help="Location of power spectra for each micrograph", gui=dict(filetype="open"))
    shrgroup.add_option("", reference_file="cluster/data/reference.dat",           help="Location of generated reference", gui=dict(filetype="open"))
    shrgroup.add_option("", align_file="cluster/data/data.star",                   help="Location of relion selection file", gui=dict(filetype="open"))
    shrgroup.add_option("", good_file="",                                          help="Location of cleaned up particle selection files", gui=dict(filetype="open"))
    #data/local/vicer/good/good_000000.dat
    shrgroup.add_option("", view_file="local/vicer/view/view_000000.dat",          help="Location of images embedded in low-dimensional factor space", gui=dict(filetype="open"))
    shrgroup.add_option("", selection_file="local/screen/select.dat",              help="Location of micrograph selection", gui=dict(filetype="open"))
    shrgroup.add_option("", small_micrograph_file="local/mic_sm/mic_000000.dat",   help="Location of micrograph selection", gui=dict(filetype="open"))
    shrgroup.add_option("", translation_file="local/movie/shift/shift_000000.dat", help="Location of frame shifts for each micrograph", gui=dict(filetype="open"))
    
    #shrgroup.add_option("", local_temp="data/temp/",                                    help="Location for temporary files (on local node for MPI)", gui=dict(filetype="open"))
    
    ########################################################################################################
    # Depreciated Parameters for Future Versions
    ########################################################################################################
    shrgroup.add_option("", data_ext="dat",                                        help="SPIDER extension")
    shrgroup.add_option("", spider_path="/guam.raid.cluster.software/spider.21.00",help="SPIDER executable", gui=dict(filetype="open"), required=True)
    pgroup.add_option_group(shrgroup)
    # create suffix system? internal database?
    
def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    #spider_params.check_options(options) # interactive
    #if len(options.ext) != 3: raise OptionValueError, "SPIDER extension must be three characters"
    
    #Move check to movie mode
    if options.gain_file == "" and len(options.input_files) > 0 and ndimage_file.count_images(options.input_files[0]) > 1:
        _logger.warn("No gain reference specified for movie mode alignment!")
    
    path = spider.determine_spider(options.spider_path)
    if path == "": options.spider_path = os.path.dirname(sys.argv[0])
    path = spider.determine_spider(options.spider_path)
    if path == "": raise OptionValueError, "Cannot find SPIDER executable in %s, please use --spider-path to specify"%options.spider_path
    options.spider_path = path
    if options.apix == 0.0:
        raise OptionValueError, "No pixel size in angstroms specified (--apix), either specifiy it or an existing SPIDER params file"
    if options.voltage == 0.0:
        raise OptionValueError, "No electron energy in KeV specified (--voltage), either specifiy it or an existing SPIDER params file"
    if options.cs == 0.0:
        raise OptionValueError, "No spherical aberration in mm specified (--cs), either specifiy it or an existing SPIDER params file"
    if options.particle_diameter == 0.0:
        raise OptionValueError, "No longest diameter of the particle in angstroms specified (--particle-diameter), either specifiy it or an existing SPIDER params file"
    if options.mask_diameter == 0.0: options.mask_diameter = options.particle_diameter*1.1

def change_option_defaults(parser):
    ''' Change the values to options specific to the script
    '''
    
    parser.change_default(log_file="local/log/project.log")

def flags():
    ''' Get flags the define the supported features
    
    :Returns:
    
    flags : dict
            Supported features
    '''
    
    return dict(description = '''Generate all the scripts and directories for an Arachnid workflow
                                 
                                 $ ara-project micrograph_files* -r raw-reference --apix 1.2 --voltage 300 --cs 2.26 --pixel-diameter 220
                              ''',
                supports_MPI=False, 
                supports_OMP=False,
                use_version=False)

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__)

if __name__ == "__main__": main()


