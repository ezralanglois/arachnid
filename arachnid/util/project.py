''' Create a project workflow

This script generates a workflow to process data collected by single-particle
cryo-electron microscopy. It generates a set of configuration files that can
be used to tune individual parameters as well as a directory structure
to organize all data.

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
    |     - autoclean.cfg
    |     - relion_selection.cfg
    |     
    |    data/
    |     - cluster/
    |         - data/
    |             - data.star
    |             - reference.dat
    |         - win/
    |             - win_00001.dat
    |             - win_00002.dat
    |     - local/
    |         - coords/
    |             - sndc_00001.dat
    |             - sndc_00002.dat
    |         - ctf/
    |             - pow/
    |                - pow_000001.dat
    |                - pow_000002.dat
    |             - params.dat
    |             - ctf.dat
    |         - mics_small/
    |         - mics/
    |    run.sh


The workflow runs the following scripts in order:

    #. project
    #. enumerate_files  (optional)
    #. reference (optional)
    #. align_frames  (optional)
    #. Defocus
    #. AutoPick
    #. Crop
    #. Refine
    #. ViCer
    #. selrelion
    #. Crop_frames

Single master config file - link input/output
Individual files - configure each program

.. Created on Oct 20, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.image import ndimage_file
import logging, os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, **extra):
    '''
    1. Write out config files
    2. Write launch scripts
        a. detect MPI
        b. download reference
        c. get data from leginon
    '''
    
    workflow = build_workflow(files, **extra)
    _logger.info("Work flow includes %d steps"%len(workflow))
    print workflow
    scripts = write_config(workflow, **extra)
    
    # Use found to determine starting point:
    #    1. enumfiles
    #    2. align_frames
    #    3. defocus
    scripts = [scripts[0]]+build_dependency_tree(scripts[1:], scripts[0])
    print [s[1] for s in scripts]

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
    subset=[]
    outdeps = set(root[3]+found)
    found=[]
    for script in scripts:
        intersect=outdeps&set(script[2])
        if len(intersect)==len(script[2]):
            branch.append(script)
        else:
            subset.append(script)
            found.extend(intersect)
    if len(subset) > 0:
        for script in branch:
            branch.extend(build_dependency_tree(subset, script, found))
    return branch

def write_workflow():
    '''
    run
    screen
    vicer
    selrelion
    '''
    
    pass
    
def write_config(workflow, **extra):
    ''' Write configurations files for each script in the
    workflow.
    
    :Parameters:
    
    workflow : list
               List of modules
    extra : dict
            Keyword arguments
    
    :Returns:
    
    scripts : list
              List of tuples containing: module, script name, 
              input file deps, output file deps.
    '''
    
    config_path=extra['config_path']
    if not os.path.exists(config_path):
        try: os.makedirs(config_path)
        except: pass
    
    scripts=[]
    for mod in workflow:
        config = program.update_config(mod, ret_file_deps=True, **extra)
        if config == "":
            config = program.write_config(mod, ret_file_deps=True, **extra)
        scripts.append((mod, )+config)
    return scripts

def build_workflow(files, **extra):
    ''' Collect all modules that belong to the current workflow
    
    This modules searchs specific packages that contain application
    scripts and includes those that meet the following criterion:
        
        #. Must have a :py:func:`supports` function
        #. And this function must return true for the given parameters
    
    :Parameters:
    
    files : list
            List of input image files
    extra : dict
            Keyword arguments
    
    :Returns:
    
    workflow : list
               List of programs
    '''
    
    import project
    import pkgutil
    from .. import app, pyspider, util
    
    workflow = [project]
    for pkg in [app, pyspider, util]:
        modules = [name for _, name, ispkg in pkgutil.iter_modules([os.path.dirname(cpkg.__file__) for cpkg in [pkg]] ) if not ispkg]
        for name in modules:
            try:
                mod = getattr(__import__(pkg.__name__, globals(), locals(), [name]), name)
            except:pass
            else:
                if hasattr(mod, 'supports') and mod.supports(files, **extra):
                    workflow.append(mod)
    return workflow

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import OptionGroup
        
    pgroup.add_option("-i", input_files=[],     help="List of input filenames containing raw micrographs or stacks of micrograph frames", required_file=True, gui=dict(filetype="open"))
    pgroup.add_option("-r", raw_reference="",   help="Raw reference volume - optional", gui=dict(filetype="open"), required=False)
    pgroup.add_option("-g", gain_reference="",  help="Gain reference image for movie mode - optional", gui=dict(filetype="open"), required=False)
    pgroup.add_option("", is_film=False,        help="Set true if the micrographs have already been contrast inverted, usually when collected on film", required=True)
    pgroup.add_option("", apix=0.0,             help="Pixel size, Angstroms", gui=dict(minimum=0.0, decimals=4, singleStep=0.1), required=True)
    pgroup.add_option("", voltage=0.0,          help="Electron energy, KeV", gui=dict(minimum=0.0, singleStep=1.0), required=True)
    pgroup.add_option("", particle_diameter=0,  help="Longest diameter of the particle, Angstroms", gui=dict(minimum=0), required=True)
    pgroup.add_option("", cs=0.0,               help="Spherical aberration, mm", gui=dict(minimum=0.0, decimals=2), required=True)
    
    shrgroup = OptionGroup(parser, "Metadata", "Files created during workflow")
    shrgroup.add_option("", config_path="cfg", help="", gui=dict(filetype="open")) # dir-open
    shrgroup.add_option("", micrograph_files="data/local/mics/mic_00000.dat", help="", gui=dict(filetype="save")) # create soft link or set of softlinks?
    shrgroup.add_option("", param_file="data/local/ctf/params.dat", help="", gui=dict(filetype="save"))
    shrgroup.add_option("", movie_files="", help="", gui=dict(filetype="file-list")) # create soft link or set of softlinks?
    shrgroup.add_option("", coordinate_file="data/local/coords/sndc_000000.dat", help="", gui=dict(filetype="open"))
    shrgroup.add_option("", particle_file="data/cluster/win/win_000000.dat", help="", gui=dict(filetype="open"))
    shrgroup.add_option("", ctf_file="data/local/ctf/ctf.dat", help="", gui=dict(filetype="open"))
    shrgroup.add_option("", reference_file="data/cluster/reference.dat", help="", gui=dict(filetype="open"))
    shrgroup.add_option("", align_file="data/cluster/data/data.star", help="", gui=dict(filetype="open"))
    shrgroup.add_option("", good_file="data/local/vicer/good/good_000000.dat", help="", gui=dict(filetype="open"))
    shrgroup.add_option("", view_file="data/local/vicer/view/view_000000.dat", help="", gui=dict(filetype="open"))
    shrgroup.add_option("", select_file="data/local/screen/select.dat", help="", gui=dict(filetype="open"))
    shrgroup.add_option("", frame_shift_file="data/local/movie/shift/shift_000000.dat", help="", gui=dict(filetype="open"))
    shrgroup.add_option("", gain_file="data/local/gain/gain.dat", help="", gui=dict(filetype="open")) # required if input is frames, 
    pgroup.add_option_group(shrgroup)
    # create suffix system? internal database?
    
def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    #spider_params.check_options(options) # interactive
    #if len(options.ext) != 3: raise OptionValueError, "SPIDER extension must be three characters"
    
    if options.gain_reference == "" and len(options.input_files) > 0 and ndimage_file.count_images(options.input_files[0]) > 1:
        _logger.warn("No gain reference specified for movie mode alignment!")
        
    if options.apix == 0.0:
        raise OptionValueError, "No pixel size in angstroms specified (--apix), either specifiy it or an existing SPIDER params file"
    if options.voltage == 0.0:
        raise OptionValueError, "No electron energy in KeV specified (--voltage), either specifiy it or an existing SPIDER params file"
    if options.cs == 0.0:
        raise OptionValueError, "No spherical aberration in mm specified (--cs), either specifiy it or an existing SPIDER params file"
    if options.particle_diameter == 0.0:
        raise OptionValueError, "No longest diameter of the particle in angstroms specified (--particle-diameter), either specifiy it or an existing SPIDER params file"

def supports2(files, **extra):
    ''' Test if this module is required in the project workflow
    
    :Parameters:
    
    files : list
            List of filenames to test
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    flag : bool
           True if this module should be added to the workflow
    '''
    
    return True

def flags():
    ''' Get flags the define the supported features
    
    :Returns:
    
    flags : dict
            Supported features
    '''
    
    return dict(description = '''Generate all the scripts and directories for an Arachnid workflow
                                 
                                 $ %prog micrograph_files* -r raw-reference --apix 1.2 --voltage 300 --cs 2.26 --pixel-diameter 220
                              ''',
                supports_MPI=False, 
                supports_OMP=False,
                use_version=False)

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__)

if __name__ == "__main__": main()


