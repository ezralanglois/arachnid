''' Classify 2D projections into view classes

.. Created on Jun 5, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''


from ..core.app import program
'''
from ..core.image import ndimage_file, eman2_utility, analysis, ndimage_utility, rotate
from ..core.metadata import spider_utility, format, format_utility, spider_params
from ..core.parallel import mpi_utility, openmp
, numpy, os, scipy, itertools, scipy.cluster.vq, scipy.spatial.distance
'''
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, **extra):
    '''Concatenate files and write to a single output file
        
    :Parameters:
    
    filename : str
               Name of the input file
    output : str
             Filename for output file
    id_len : int
             Max length of SPIDER ID
    extra : dict
            Unused key word arguments
                
    :Returns:
    
    filename : str
               Current filename
    '''
    
    pass

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Classify 2D", "Options to control reference-free 2D classification",  id=__name__)
    group.add_option("", nsamples=70,                 help="Number of rotational samples")
    group.add_option("", cache_file="",               help="Cache preprocessed data in matlab data files")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with with no digits at the end (e.g. this is bad -> sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)

def check_options(options, main_option=False):
    #Check if the option values are valid
#    from ..core.app.settings import OptionValueError

    if options.nsamples < 1: options.nsamples=1

def main():
    #Main entry point for this script
    program.run_hybrid_program(__name__,
        description = '''Classify particles into 2D views
                        
                        http://
                        
                        Example:
                         
                        $ ara-classify2d input-stack.spi -o align.dat -p params.spi
                      ''',
        use_version = True,
        supports_OMP=True,
    )

#def dependents(): return [spider_params]
if __name__ == "__main__": main()

