''' Delete files based on a selection file

.. Created on Apr 16, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.metadata import format
from ..core.metadata import spider_params
from ..core.metadata import selection_utility
import logging
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, follow_link=False, **extra):
    '''
    '''
    
    if follow_link and os.path.islink(filename):
        link = os.readlink(filename)
        os.unlink(link)
    os.unlink(filename)
    return filename

def initialize(files, param):
    # Initialize global parameters for the script
    
    if not os.path.exists(param['selection_file']):
        raise ValueError, "Requires existing selection file"
    select = format.read(param['selection_file'], numeric=True)
    oldcnt = len(files)
    files = selection_utility.select_file_subset(files, select, param.get('id_len', 0))
    _logger.info("Deleting %d Files of %d"%(len(files), oldcnt))
    _logger.info("Example files include:")
    for filename in files[:5]:
        _logger.info("  -   %s"%filename)
    if param['yes']: return files
    val = str(input('Are you sure you want to delete these files? (Yes or No)')).strip()
    if len(val) == 0 or val.lower()[0]!='y': return []
    return files

def finalize(files, **extra):
    # Finalize global parameters for the script
    
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Delete", "Options to control frame alignment",  id=__name__)
    group.add_option("", follow_link=False,   help="Delete link and original file")
    group.add_option("", yes=False,           help="Do not ask whether the user is sure")
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[],           help="List of filenames for the input micrographs, e.g. mic_*.mrc", required_file=True, gui=dict(filetype="open"))
        pgroup.add_option("-s",   selection_file="",        help="Selection file", gui=dict(filetype="open"), required_file=True)
        spider_params.setup_options(parser, pgroup, False)

def flags():
    ''' Get flags the define the supported features
    
    :Returns:
    
    flags : dict
            Supported features
    '''
    
    return dict(description = '''Delete selected file
                        
                        Example: Unprocessed film micrograph
                         
                        $ ara-delete mic_0000.dat -s select.dat
                      ''',
                supports_MPI=False, 
                supports_OMP=True,
                use_version=True)

def main():
    #Main entry point for this script
    program.run_hybrid_program(__name__)

if __name__ == "__main__": main()

