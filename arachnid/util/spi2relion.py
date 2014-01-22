''' Generate a single aligned stack from a relion selection file

.. Created on Apr 22, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.metadata import format, spider_utility, spider_params
from ..core.orient import spider_transforms
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, voltage, cs, ampcont, stack_file="", **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
    
    files : list
            List of input filenames
    output : str
             Output filename for alignment file
    extra : dict
            Unused keyword arguments
    '''
    
    align = format.read_alignment(files[0])
    vals = []
    
    header="rlnVoltage,rlnDefocusU,rlnSphericalAberration,rlnAmplitudeContrast,rlnImageName,rlnMicrographName,rlnGroupNumber,rlnOriginX,rlnOriginY,rlnAngleRot,rlnAngleTilt,rlnAnglePsi,rlnClassNumber".split(',')
    group={}
    for val in align:
        mic = spider_utility.spider_filename(stack_file, val.micrograph)
        if mic not in group: group[mic]=len(group)+1
        psi, tx, ty = spider_transforms.align_param_2D_to_3D(val.psi, val.tx, val.ty)
        vals.append( [voltage, val.defocus, cs, ampcont, "%d@%s"%(val.stack_id, mic), mic, group[mic], tx, ty, val.phi, val.theta, psi, 1 ] )
    
    _logger.info("Writing alignment file")
    format.write(output, vals, header=header)
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    
    if pgroup is None: pgroup=parser
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input images or stacks named according to the SPIDER format", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-m", stack_file="", help="Path to stack file", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Base filename for output volume and half volumes, which will be named raw_$output, raw1_$output, raw2_$output", gui=dict(filetype="save"), required_file=True)

#def check_options(options, main_option=False):
    #Check if the option values are valid
    #from ..core.app.settings import OptionValueError
    

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Generate a relion star file from a SPIDER alignment file
                        
                        $ %prog -i align -p params -o relion_star_file.star
                        
                        
                        http://
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=False,
        use_version = False,
    )
def dependents(): return [spider_params]
if __name__ == "__main__": main()

