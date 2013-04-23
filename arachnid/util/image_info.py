''' Image information

.. Created on Apr 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.image import ndimage_file
import logging, os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output="", **extra):
    '''Generate a relion selection file
    
    :Parameters:
    
    filename : tuple
               File index and input filename
    extra : dict
            Unused key word arguments
    '''
    
    # print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x)
    for i, filename in enumerate(files):
        header = ndimage_file.read_header(filename)
        header['name'] = os.path.basename(filename)
        if i == 0:
            header_name = dict([(k, k) for k in header.iterkeys()])
            print '{name:50}  {nx:3} {ny:3} {nz:3} {count:7} {apix:7}'.format(**header_name)
        print '{name:50} {nx:3d} {ny:3d} {nz:3d} {count:7d} {apix:7.2f}'.format(**header)
    _logger.info("Complete")

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    #from ..core.app.settings import OptionGroup
    #group = OptionGroup(parser, "Image information", "Options to view image information",  id=__name__)
    #group.add_option("-f", full=False,                   help="Print full header")
    #pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input stacks or selection file", required_file=True, gui=dict(filetype="file-list"))
        #pgroup.add_option("-o", output="",      help="Output filename for the relion selection file", gui=dict(filetype="save"), required_file=True)
        parser.change_default(log_level=3)

#def check_options(options, main_option=False):
    #Check if the option values are valid
#    from ..core.app.settings import OptionValueError


def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Image information
        
                         http://
                         
                         Example:
                         
                         $ ara-info image.spi
                      ''',
        supports_MPI = False,
        use_version = False,
    )
def dependents(): return []
if __name__ == "__main__": main()



