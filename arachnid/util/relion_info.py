''' Summarize a Relion run


.. Created on Jan 30, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
#from ..core.metadata import spider_utility, format_utility, spider_params, selection_utility
from ..core.metadata import format
from ..core.metadata import spider_utility
#from ..core.metadata import relion_utility
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, **extra):
    '''Crop a set of particles from a micrograph file with the specified
    coordinate file and write particles to an image stack.
    
    :Parameters:
    
        filename : str
                   Input filename
        extra : dict
                Unused keyword arguments
                
    :Returns:
            
        val : string
              Current filename
    '''
    
    data = format.read(filename)
    _logger.info("Found %d tables"%len(data))
    for key, d in data.iteritems():
        _logger.info("Table (%s): %s"%(key, ",".join(d[0]._fields)))
    return filename, None

def initialize(files, param):
    ''' Initalize the file processor
    
    :Parameters:
        
        files : list
                List of input files
        param : dict
                Keyword arguments from the user
    
    :Returns:
    
        files : list
                List of files to process        
    '''
    
    _logger.info("Extracting columns: %s"%",".join(param['model_columns']))
    return files

def reduce_all(val, table=None, **extra):
    '''
    '''

    filename, row = val
    return str(filename)

def finalize(files, table=None, **extra):
    '''
    '''
    
    _logger.info("Completed")
    
def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    
    group = OptionGroup(parser, "Cropping", "Options to crop particles from micrographs", id=__name__) #, gui=dict(root=True, stacked="prepComboBox"))
    group.add_option("", model_columns=['data_model_classes@rlnClassDistribution'],  help="Attributes to extract form a Relion model file")    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[],    help="List of input filenames written out by Relion", required_file=True, gui=dict(filetype="open"), regexp=spider_utility.spider_searchpath)
        pgroup.add_option("-o", output="",         help="Output filename for summary", gui=dict(filetype="save"), required_file=True)
        parser.change_default(log_level=3)

#def check_options(options, main_option=False):
    #Check if the option values are valid
    
#    from ..core.app.settings import OptionValueError


def flags():
    ''' Get flags the define the supported features
    
    :Returns:
    
    flags : dict
            Supported features
    '''
    
    return dict(description = '''Summarize the output of a Relion run
                         
                        Example: Run from the command line on a single node
                        
                        $ %prog run1/relion_it003_data.star -o summary.html
                      ''',
                supports_MPI=False, 
                supports_OMP=False,
                use_version=False)

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__)

def dependents(): return []

if __name__ == "__main__": main()


