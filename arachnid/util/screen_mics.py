''' Preprocess a set of micrographs for screening

.. Created on Feb 14, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.metadata import format, spider_utility, spider_params #, format_utility, format, spider_params
from ..core.image import ndimage_file, ndimage_utility, eman2_utility
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, bin_factor, sigma, film, clamp_window, window=0, id_len=0, **extra):
    '''Concatenate files and write to a single output file
        
    :Parameters:
    
    filename : str
               Input filename
    id_len : int, optional
             Maximum length of the ID
    extra : dict
            Unused key word arguments
                
    :Returns:
    
    filename : str
               Current filename
    peaks : str
            Coordinates found
    '''
    
    output = spider_utility.spider_filename(output, filename, id_len)
    mic = ndimage_file.read_image(filename)
    if bin_factor > 1.0:
        mic = eman2_utility.decimate(mic, bin_factor)
    if sigma > 0.0 and window > 0:
        mic = eman2_utility.gaussian_high_pass(mic, sigma/(window), True)
        if eman2_utility.is_em(mic): 
            emic=mic
            mic = eman2_utility.em2numpy(emic).copy()
    if not film: ndimage_utility.invert(mic, mic)
    ndimage_utility.replace_outlier(mic, clamp_window, out=mic)
    ndimage_file.write_image(output, mic)
    return filename

def initialize(files, param):
    # Initialize global parameters for the script
    
    _logger.info("Window size: %d"%(param['window']))
    if param.get('window', 0) == 0:
        _logger.info("High-pass filtering disabled - no params file")
    if param['sigma'] > 0 and param.get('window', 0)>0: _logger.info("High pass filter: %f"%(param['sigma'] / param['window']))
    if param['bin_factor'] > 1: _logger.info("Decimate micrograph by %d"%param['bin_factor'])
    if not param['film']: _logger.info("Inverting contrast of the micrograph")
    _logger.info("Dedust: %f"%param['clamp_window'])
    
    if param['select'] != "":
        select = format.read(param['select'], numeric=True)
        files = spider_utility.select_subset(files, select, param.get('id_len', 0))
    return files

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Screen", "Options to control screening images",  id=__name__)
    group.add_option("-s", select="",              help="SPIDER micrograph or class selection file - if select file does not have proper header, then use `--select filename=id` or `--select filename=id,select`", gui=dict(filetype="open"))
    group.add_option("", clamp_window=2.5,         help="Number of standard deviations to replace extreme values using a Gaussian distribution (0 to disable)")
    group.add_option("", sigma=1.0,                help="Highpass factor: 1 or 2 where 1/window size or 2/window size (0 to disable)")
    group.add_option("", film=False,               help="Do not invert the contrast on the micrograph (inversion is generally done during scanning for film)")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input stacks or selection file", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the relion selection file", gui=dict(filetype="save"), required_file=True)
    
def setup_main_options(parser, pgroup=None):
    #
    parser.change_default(log_level=3, bin_factor=8.0)
    

#def check_options(options, main_option=False):
    #Check if the option values are valid
#    from ..core.app.settings import OptionValueError


def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Prepare a set of micrographs for screening
        
                         http://
                         
                         Example: Generate a relion selection file from a set of stacks, defocus file and params file
                         
                         $ ara-screen mic*.mrc -o mic_00000.spi -p params.spi
                      ''',
        supports_MPI = False,
        use_version = False,
    )

def dependents(): return [spider_params]
if __name__ == "__main__": main()



