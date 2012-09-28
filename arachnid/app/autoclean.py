'''
.. Created on Sep 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.image import ndimage_file, eman2_utility, analysis, ndimage_utility
from ..core.metadata import spider_utility, format, format_utility
from ..core.parallel import mpi_utility
import logging, numpy, os #, scipy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(input_vals, input_files, output, **extra):
    '''Concatenate files and write to a single output file
        
    :Parameters:
    
    input_vals : list 
                 Tuple(view id, image labels and alignment parameters)
    input_files : list
                  List of input file stacks
    output : str
             Filename for output file
    extra : dict
            Unused key word arguments
                
    :Returns:
    
    filename : str
               Current filename
    peaks : str
            Coordinates found
    '''
    
    data = read_data(input_files, *input_vals[1:], **extra)
    eigs, sel = classify_data(data, **extra)
    if eigs.ndim == 1: eigs = eigs.reshape((eigs.shape[0], 1))
    feat = numpy.hstack((sel[:, numpy.newaxis], eigs))
    format.write(os.path.splitext(output)[0]+".csv", feat, prefix="embed_", spiderid=input_vals[0])
    return input_vals[0], sel

def classify_data(data, **extra):
    ''' Classify the aligned projection data grouped by view
    
    :Parameters:
    
    data : array
           2D array where each row is an aligned and transformed image
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    eigs : array
           2D array of embedded images
    sel : array
          1D array of selected images
    '''
    
    eigs = analysis.pca(data, frac=1)[0] #val, idx, V[:idx], t[idx]
    sel = analysis.one_class_classification(eigs)
    return eigs, sel

def image_transform(img, align, mask, use_rtsq=False, resolution=0.0, apix=None, **extra):
    ''' Transform an image
    
    .. todo:: add ctf correction
    
    .. todo:: add bispectrum here
    
    :Parameters:
    
    img : array
          Image data
    align : array
            Alignment parameters
    mask : array
           Mask
    use_rtsq : bool
               Set true to rotate and translate
    resolution : float
                 Resolution to filter data
    apix : float
           Pixel size
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    out : array
          1D transformed image
    '''
    
    m = align[1] > 180.0
    if use_rtsq: img = eman2_utility.rot_shift2D(img, align[5], align[6], align[7], m)
    else:        img = eman2_utility.mirror(img)
    if resolution > 0:
        img = eman2_utility.gaussian_low_pass(img, apix/resolution, True)
        bin_factor = min(8, resolution / apix*2)
    img = eman2_utility.decimate(img, bin_factor)
    if 1 == 0:
        pass # add bispectrum
    else:
        ndimage_utility.normalize_standard(img, mask, True, img)
        img = ndimage_utility.compress_image(img, mask)
    return img

def read_data(input_files, label, align, pixel_diameter=None, **extra):
    ''' Read images from a file and transform into a matrix
    
    :Parameters:
    
    input_files : list
                  List of input file stacks
    label : array
            2D array of particle labels (stack_id, particle_id)
    align : array
            2D array of alignment parameters
    pixel_diameter : int
                     Diameter of the particle in pixels
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    data : array
           2D array of transformed image data, each row is a transformed image
    '''
    
    mask = None
    data = None
    label[:, 1]-=1
    for i, img in enumerate(ndimage_file.iter_images(input_files, label)):
        if mask is None:
            mask = ndimage_utility.model_disk(int(pixel_diameter/2), img.shape)
        img = image_transform(img, align[i], mask, **extra)
        if data is None:
            data = numpy.zeros((label.shape[0], img.ravel().shape[0]))
        data[i, :] = img.ravel()
    return data

def read_alignment(files, alignment="", **extra):
    ''' Read alignment parameters
    
    :Parameters:
    
    files : list
            List of input files containing particle stacks
    alignment : str
                Input filename containing alignment parameters
    
    :Returns:
    
    group : list
            List of tuples, one for each view containing: view id, image labels and alignment parameters
    '''
    
    if len(files) == 1:
        spiderid = files[0] if not os.path.exists(alignment) else None
        align = format.read_alignment(alignment, spiderid=spiderid)
        align, header = format_utility.tuple2numpy(align)
        refidx = header.index('ref_num')
        label = numpy.zeros((len(align), 2), dtype=numpy.int)
        label[:, 0] = spider_utility.spider_id(files[0])
        label[:, 1] = align[:, 4].astype(numpy.int)
        if numpy.max(label[:, 1]) > ndimage_file.count_images(files[0]):
            label[:, 1] = numpy.arange(1, len(align)+1)
    else:
        align = None
        refidx = None
        if os.path.exists(alignment):
            _logger.debug("Alignment exists")
            align = format.read_alignment(alignment)
            align, header = format_utility.tuple2numpy(align)
            refidx = header.index('ref_num')
            if len(align)>0 and 'stack_id' in set(header):
                align = numpy.asarray(align)
                label = align[:, 15:17].astype(numpy.int)
            else:
                align=None
        if align is None:
            alignvals = []
            total = 0 
            for f in files:
                aligncur = format.read_alignment(alignment, spiderid=f)
                aligncur, header = format_utility.tuple2numpy(aligncur)
                if refidx is None: refidx = header.index('ref_num')
                alignvals.append((f, aligncur))
                total += len(aligncur)
            label = numpy.zeros((total, 2), dtype=numpy.int)
            align = numpy.zeros((total, aligncur.shape[1]))
            total = 0
            for f, cur in alignvals:
                end = total+cur.shape[0]
                align[total:end, :] = cur
                label[total:end, 0] = spider_utility.spider_id(f)
                label[total:end, 1] = align[total:end, 4].astype(numpy.int)
                if numpy.max(label[total:end, 1]) > ndimage_file.count_images(f):
                    label[:, 1] = numpy.arange(1, len(align[total:end])+1)
                total = end
    ref = align[:, refidx].astype(numpy.int)
    refs = numpy.unique(ref)
    
    group = []
    for r in refs:
        sel = r == ref
        group.append((r, label[sel], align[sel]))
    return group

def initialize(files, param):
    # Initialize global parameters for the script
    
    if mpi_utility.is_root(**param):
        if param['resolution'] > 0.0: _logger.info("Filter and decimate to resolution: %f"%param['resolution'])
        if param['use_rtsq']: _logger.info("Rotate and translate data stack")
    
    group = None
    if mpi_utility.is_root(**param): group = read_alignment(files, **param)
    return mpi_utility.broadcast(group, **param)

def reduce_all(val, **extra):
    # Process each input file in the main thread (for multi-threaded code)

    # Write average
    pass

def finalize(files, **extra):
    # Finalize global parameters for the script
    
    # 1. Write total, selected, tossed
    # 2. Write selection file by micrograph
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "AutoPick", "Options to control reference-free particle selection",  id=__name__)
    group.add_option("", resolution=20.0,       help="Filter to given resolution - requires apix to be set")
    group.add_option("", use_rtsq=False,        help="Use alignment parameters to rotate projections in 2D")
    group.add_option("", neig=1.0,              help="Number of eigen vectors to use")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-a", alignment="",   help="Input file containing alignment parameters", required_file=True, gui=dict(filetype="open"))
def check_options(options, main_option=False):
    #Check if the option values are valid
    
    #from ..core.app.settings import OptionValueError
    pass

def main():
    #Main entry point for this script
    run_hybrid_program(__name__,
        description = '''Clean a particle selection of any remaning bad windows
                        
                        http://
                        
                        Example:
                         
                        $ ara-autoclean input-stack.spi -o coords.dat -p params.spi
                      ''',
        use_version = True,
        supports_OMP=True,
    )

def dependents(): return []
if __name__ == "__main__": main()

