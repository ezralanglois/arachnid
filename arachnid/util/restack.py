''' Restack a set of images

.. Created on Feb 14, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.metadata import format, spider_utility #, format_utility, format, spider_params
from ..core.image import ndimage_file
import logging,os,numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, select="", id_len=0, unstack=False, stack=False, average=False, compare_file="", **extra):
    '''Generate a relion selection file
    
    :Parameters:
    
    filename : tuple
               File index and input filename
    extra : dict
            Unused key word arguments
    '''
    
    if unstack:
        index = 1
        for filename in files:
            for img in ndimage_file.iter_images(filename):
                ndimage_file.write_image(spider_utility.spider_filename(output, index), img)
                index += 1
        
        return
    if stack:
        index = 0
        selected=None
        tot = ndimage_file.count_images(files)
        _logger.info("Stacking %d files with %d images"%(len(files), tot))
        label = numpy.zeros((tot, 2))
        for filename in files:
            if select != "":
                if not os.path.exists(spider_utility.spider_filename(select, id)):
                    _logger.warn("No selection file for %d"%id)
                    continue
                selected = [s.id-1 for s in format.read(select, numeric=True, spiderid=id)]
            tot = ndimage_file.count_images(filename) if selected is None else len(selected)
            label[index:index+tot, 0] = spider_utility.spider_id(filename)
            label[index:index+tot, 1] = numpy.arange(1, tot+1)
            if compare_file != "":
                cmp1 = ndimage_file.read_image(compare_file, index)
                cmp2 = ndimage_file.read_image(filename)
                diff = numpy.sqrt(numpy.sum(numpy.square(cmp1-cmp2)))
                _logger.error("here: %f"%diff)

            for img in ndimage_file.iter_images(filename, selected):
                ndimage_file.write_image(output, img, index)
                index += 1
        format.write(output, label, prefix='sel_', header="mic,par".split(','))
        return
    
    if average:
        avg = None
        total = 0
        for filename in files:
            id = spider_utility.spider_id(filename, id_len)
            _logger.info("Averaging %d with %d images"%(id, ndimage_file.count_images(filename)))
            for index, img in enumerate(ndimage_file.iter_images(filename)):
                if avg is None:
                    avg = img.copy()
                else:
                    avg += img
                total += 1
        
        ndimage_file.write_image(output, avg/total)
        return
    
    for filename in files:
        id = spider_utility.spider_id(filename, id_len)
        output = spider_utility.spider_filename(output, id)
        if select != "":
            if not os.path.exists(spider_utility.spider_filename(select, id)):
                _logger.warn("No selection file for %d"%id)
                continue
            selected = [s.id-1 for s in format.read(select, numeric=True, spiderid=id)]
            _logger.info("Restacking %d with %d selections of %d images"%(id, len(selected), ndimage_file.count_images(filename)))
            for index, img in enumerate(ndimage_file.iter_images(filename, selected)):
                ndimage_file.write_image(output, img, index)
        else:
            _logger.info("Restacking %d with %d images"%(id, ndimage_file.count_images(filename)))
            for index, img in enumerate(ndimage_file.iter_images(filename)):
                ndimage_file.write_image(output, img, index)
    _logger.info("Complete")

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Restack", "Options to control creation of stack files",  id=__name__)
    group.add_option("-s", select="",                       help="SPIDER micrograph or class selection file - if select file does not have proper header, then use `--select filename=id` or `--select filename=id,select`", gui=dict(filetype="open"))
    group.add_option("-u", unstack=False,                   help="Unstack single images into a stack")
    group.add_option("-x",  stack=False,                    help="Stack single images into a stack")
    group.add_option("-m", average=False,                    help="Average a set of images/stacks into a single image")
    group.add_option("", compare_file="",                    help="Compare to another set of images")
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input stacks or selection file", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the relion selection file", gui=dict(filetype="save"), required_file=True)
        parser.change_default(log_level=3)

#def check_options(options, main_option=False):
    #Check if the option values are valid
#    from ..core.app.settings import OptionValueError


def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Restack a set of images
        
                         http://
                         
                         Example: Generate a relion selection file from a set of stacks, defocus file and params file
                         
                         $ ara-restack "win*.spi" -s good_00000.spi -o win_good_0000.spi
                      ''',
        supports_MPI = False,
        use_version = False,
    )
def dependents(): return []
if __name__ == "__main__": main()



