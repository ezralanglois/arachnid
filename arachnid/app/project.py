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
    |     - rename2spi.cfg
    |     - reference.cfg
    |     - frame_align.cfg
    |     - estimate_ctf.cfg
    |     - autopick.cfg
    |     - crop.cfg
    |     - autorefine.cfg
    |     - autoclean.cfg
    |     - relion_selection.cfg
    |     - global.cfg
    |     
    |    data/
    |     - reconstruct/
    |         - data.star
    |         - win
    |         - params.dat
    |     - preprocess/
    |         - coords/
    |         - pow/
    |         - mics_small/
    |         - mics_avg/
    |    run.sh


The workflow runs the following scripts in order:

    #. Rename2spi  (optional)
    #. Project
    #. Reference (optional)
    #. Alignment  (optional)
    #. Defocus
    #. AutoPick
    #. Crop
    #. Refine
    #. ViCer
    #. selrelion

Single master config file - link input/output
Individual files - configure each program

.. Created on Oct 20, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, **extra):
    '''
    '''
    
    pass

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    #from ..core.app.settings import OptionGroup
        
    pgroup.add_option("-i", input_files=[],     help="List of input filenames containing raw micrographs or stacks of micrograph frames", required_file=True, gui=dict(filetype="file-list"))
    pgroup.add_option("-r", raw_reference="",   help="Raw reference volume - optional", gui=dict(filetype="open"), required=False)
    pgroup.add_option("", is_film=False,        help="Set true if the micrographs have already been contrast inverted, usually when collected on film", required=True)
    pgroup.add_option("", apix=0.0,             help="Pixel size, Angstroms", gui=dict(minimum=0.0, decimals=4, singleStep=0.1), required=True)
    pgroup.add_option("", voltage=0.0,          help="Electron energy, KeV", gui=dict(minimum=0.0, singleStep=1.0), required=True)
    pgroup.add_option("", particle_diameter=0,  help="Longest diameter of the particle, Angstroms", gui=dict(minimum=0), required=True)
    pgroup.add_option("", cs=0.0,               help="Spherical aberration, mm", gui=dict(minimum=0.0, decimals=2), required=True)
    
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
    )
if __name__ == "__main__": main()


