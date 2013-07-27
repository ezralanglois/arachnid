''' Image information

.. Created on Apr 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.image import ndimage_file
import logging, os, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output="", test_size=0, all=False, stat=False, force=False, **extra):
    '''Generate a relion selection file
    
    :Parameters:
    
    filename : tuple
               File index and input filename
    extra : dict
            Unused key word arguments
    '''
    
    # print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x)
    size = (None, None, None)
    for i, filename in enumerate(files):
        if test_size == 2:
            count = ndimage_file.count_images(filename)
            for j in xrange(count):
                header = ndimage_file.read_header(filename, j)
                if size[0] is None:
                    size = (header['nx'], header['ny'], header['nz'])
                if header['nx'] != size[0]:
                    raise ValueError, "%s - nx != nx: %d != %d"%(filename, header['nx'], size[0])
                if header['ny'] != size[1]:
                    raise ValueError, "%s - nx != nx: %d != %d"%(filename, header['ny'], size[1])
                if header['nz'] != size[2]:
                    raise ValueError, "%s - nx != nx: %d != %d"%(filename, header['nz'], size[2])
        elif test_size == 1:
            if size[0] is None:
                size = (header['nx'], header['ny'], header['nz'])
            if header['nx'] != size[0]:
                raise ValueError, "%s - nx != nx: %d != %d"%(filename, header['nx'], size[0])
            if header['ny'] != size[1]:
                raise ValueError, "%s - nx != nx: %d != %d"%(filename, header['ny'], size[1])
            if header['nz'] != size[2]:
                raise ValueError, "%s - nx != nx: %d != %d"%(filename, header['nz'], size[2])
        if force and ndimage_file.spider.is_readable(filename):
            header = ndimage_file.spider.read_header(filename)
        elif force and ndimage_file.mrc.is_readable(filename):
            header = ndimage_file.mrc.read_header(filename)
        else:
            header = ndimage_file.read_header(filename)
        header['name'] = os.path.basename(filename)
        if all:
            #count = ndimage_file.count_images(filename)
            for key, val in header.iteritems():
                print key, ": ", val
            if stat:
                #avg=0
                #std=0
                for img in ndimage_file.iter_images(filename):
                    print "Mean: ", numpy.mean(img)
                    print "STD: ", numpy.std(img)
                    print "Max: ", numpy.max(img)
                    print "Min: ", numpy.min(img)
                    print "Unique: ", len(numpy.unique(img))
                
                    #avg += numpy.mean(img)
                    #std += numpy.std(img)
                #print "Mean: ", avg/count
                #print "STD: ", std/count
        else:
            if i == 0:
                header_name = dict([(k, k) for k in header.iterkeys()])
                print '{name:50}  {nx:3} {ny:3} {nz:3} {count:7} {apix:7}'.format(**header_name)
            print '{name:50} {nx:3d} {ny:3d} {nz:3d} {count:7d} {apix:7.2f}'.format(**header)
       
    _logger.info("Complete")

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Image information", "Options to view image information",  id=__name__)
    group.add_option("-t", test_size=0,                   help="Test if the image sizes are consistent")
    group.add_option("-a", all=False,                     help="Print the entire header in long format")
    group.add_option("-s", stat=False,                    help="Estimate statistics")
    group.add_option("-f", force=False,                   help="Force internal formats")
    pgroup.add_option_group(group)
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



