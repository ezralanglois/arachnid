''' Prepare a stack of images and alignment file for bootstrapping


.. Created on Feb 21, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.metadata import spider_utility, format_utility, format, spider_params, relion_utility
from ..core.image.formats import mrc as mrc_file
from ..core.image import ndimage_file
from ..core.spider import spider
#from ..core.orient import orient_utility
import logging, numpy, os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, param_file, image_file, **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
    
    files : list
            List of input filenames
    output : str
             Output filename for alignment file
    extra : dict
            Unused keyword arguments
    '''
    
    relion_data = format.read(files[0], numeric=True)
    tmp_stack = format_utility.add_prefix(output, "tmp_")
    tmp_singe_stack = format_utility.add_prefix(output, "tmp_single_")
    #tmp2_stack = format_utility.add_prefix(output, "tmp2_")
    if not hasattr(relion_data[0], 'rlnImageName'):
        _logger.info("Processing SPIDER alignment file: %s"%files[0])
        spi = spider.open_session(files, **extra)
        spider_params.read(spi.replace_ext(param_file), extra)
        # stack phase flipped input stack
        defocus = 0
        ftimage = None
        ctfimage = None
        ctf = None
        if not os.path.exists(spi.replace_ext(tmp_stack)):
            _logger.info("Phase flipping %d particles"%len(relion_data))
            for index, align in enumerate(relion_data):
                image_file = spider_utility.spider_filename(image_file, int(align[15]))
                if defocus != float(align[17]):
                    defocus = float(align[17])
                    ctf = spi.tf_ct(defocus=defocus, outputfile=ctf, **extra)
                ftimage = spi.ft((image_file, int(align[16])), outputfile=ftimage, **extra)
                ctfimage = spi.mu(ftimage, ctf, outputfile=ctfimage)
                spi.ft(ctfimage, outputfile=(tmp_stack, index+1))
        
        align = format_utility.tuple2numpy(format.read_alignment(files[0]))[0]
        align[:, 6:8] /= extra['apix']
        align[:, 12:14] /= extra['apix']
        alignment = format.write(output, align, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,mic,slice,defocus".split(','), prefix="align_")
        #_logger.info("Running RT SQ")
        spi.rt_sq(tmp_stack, alignment, outputfile=output)
    else:
        _logger.info("Processing star file: %s"%files[0])
        filename = relion_data[0].rlnImageName
        if filename.find('@') != -1: filename = filename.split('@')[1]
        spi = spider.open_session([filename], **extra)
        spider_params.read(spi.replace_ext(param_file), extra)
        
        values = numpy.zeros((len(relion_data), 15))
        # stack phase flipped input stack
        defocus = 0
        ftimage = None
        ctfimage = None
        ctf = None
        
        
        is_mrc_flag=None
        flag = os.path.exists(spi.replace_ext(tmp_stack))
        for index, projection in enumerate(relion_data):
            if not flag:
                filename = relion_utility.relion_file(projection.rlnImageName)
                _logger.info("Flipping: %s"%projection.rlnImageName)
                if is_mrc_flag is None: is_mrc_flag = is_mrc(filename)
                if defocus != float(projection.rlnDefocusU):
                    defocus = float(projection.rlnDefocusU)
                    ctf = spi.tf_ct(defocus=defocus, outputfile=ctf, **extra)
                if is_mrc_flag:
                    offset=None
                    if isinstance(filename, tuple):
                        filename, offset = filename
                        offset-=1
                    else: filename=filename
                    img = mrc_file.read_image(filename, offset)
                    ndimage_file._default_write_format.write_spider_image(tmp_singe_stack, img)
                    filename = tmp_singe_stack
                ftimage = spi.ft(filename, outputfile=ftimage, **extra)
                ctfimage = spi.mu(ftimage, ctf, outputfile=ctfimage)
                filename=spi.ft(ctfimage, outputfile=(tmp_stack, index+1))
            else: filename = (tmp_stack, index+1)
            
            #spi.sh_f(filename, (projection.rlnOriginX, projection.rlnOriginY), outputfile=(output, index+1))
            values[index, 0] = projection.rlnAnglePsi
            #psi, dx, dy = orient_utility.align_param_3D_to_2D_simple(projection.rlnAnglePsi, projection.rlnOriginX, projection.rlnOriginY)
            values[index, 1] = projection.rlnAngleTilt
            values[index, 2] = projection.rlnAngleRot
            #values[index, 5] = psi
            #values[index, 6] = dx
            #values[index, 7] = dy
            values[index, 6] = projection.rlnOriginX
            values[index, 7] = projection.rlnOriginY

        _logger.info("Writing alignment file")
        alignment = format.write(output, values, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror".split(','), prefix="align_")
        #spi.rt_sq(tmp_stack, alignment, outputfile=output)
        #spi.rt_sq(tmp2_stack, alignment, outputfile=output, alignment_cols=(6,0,9,10))
        spi.sh_f(tmp_stack, alignment, outputfile=output)
    #spi.de(tmp_stack)
    _logger.info("Completed")

def is_mrc(filename):
    '''
    '''
    if isinstance(filename, tuple): filename=filename[0]
    
    try:
        return mrc_file.count_images(filename) > 1
    except:
        return False
    
def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import setup_options_from_doc
    
    if pgroup is None: pgroup=parser
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input images or stacks named according to the SPIDER format", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Base filename for output volume and half volumes, which will be named raw_$output, raw1_$output, raw2_$output", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-m", image_file="",  help="Image filename template for SPIDER")
        setup_options_from_doc(parser, spider.open_session, group=pgroup)
        spider_params.setup_options(parser, pgroup, True)
    
    if main_option:
        parser.change_default(thread_count=4, log_level=3)


def check_options(options, main_option=False):
    #Check if the option values are valid
    #from ..core.app.settings import OptionValueError
    
    spider_params.check_options(options)

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Prepare a stack of images and alignment file for bootstrapping
                        
                        $ %prog relion_star_file.star -p params.ter -o dala.ter
                        
                        Note that the output will be two files
                        
                        1. data.ter - aligned and phaseflipped stack
                        2. align_dala.ter - alignment parameters (3 Euler angles)
                        
                        http://
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                        exit 0
                      ''',
        supports_MPI=False,
        use_version = False,
    )
if __name__ == "__main__": main()

