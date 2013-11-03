''' Average frames with translational alignment

.. Created on Apr 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.app import program
from ..core.metadata import format, spider_utility
from ..core.image import ndimage_file, ndimage_utility
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, id_len=0, **extra):
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
    
    if isinstance(filename, tuple):
        tot = len(filename[1])
        fid = filename[0]
    else:
        try:
            tot = ndimage_file.mrc.count_images(filename)
        except: tot = ndimage_file.count_images(filename)
        fid = spider_utility.spider_id(filename, id_len)
    tot;
    spider_utility.update_spider_files(extra, fid, 'output', 'frame_align')
    
    align = format.read(extra['frame_align'], numeric=True)
    total = ndimage_file.count_images(filename[1][0])
    for index in xrange(total):
        avg = None
        for i, filename in enumerate(filename[1][1:]):
            img = ndimage_file.read_image(filename, index)
            img = ndimage_utility.fourier_shift(img, align[i].dx, align[i].dy)
            if avg is None: avg = img
            else: avg += img
        ndimage_file.write_image(extra['output'], img/len(filename[1]), index)
    return filename

def init_root(files, param):
    # 
    
    files = spider_utility.single_images(files)
    if param['reverse']:
        for f in files:
            f[1].reverse()
    _logger.info("Processing %d micrographs"%len(files))
    _logger.info("Reverse: %d"%param['reverse'])
    if len(files[0][1]) == 1: raise ValueError, "Cannot average single frame"
    return files

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Average frame", "Options to average frames",  id=__name__)    
    group.add_option("-a", frame_align="",           help="Translational alignment parameters for individual frames", required_file=True, gui=dict(filetype="file-open"))
    group.add_option("-r", reverse=False,           help="Reverse the order of the average")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input stacks or selection file", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the relion selection file", gui=dict(filetype="save"), required_file=True)

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Average movie frames from a direct director
        
                         http://
                         
                         Example: Generate a relion selection file from a set of stacks, defocus file and params file
                         
                         $ ara-frameavg frame_1_*.spi -o avg_00000.spi -a trans_0000.spi
                      ''',
        supports_MPI = False,
        use_version = False,
    )

def dependents(): return []
if __name__ == "__main__": main()



