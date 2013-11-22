''' Classify a set of projections using a Deep Learning Algorithm

.. Created on Dec 26, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.app import program
from ..core.image import ndimage_file, ndimage_interpolate, ndimage_utility, ndimage_processor #, analysis
from ..core.metadata import spider_utility, format, format_utility, spider_params
import logging, numpy, os, sys #, itertools, sys, scipy
from ..core.learn.deep import SdA

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, train_select="", **extra):
    '''
    '''
    
    label, align = read_alignment(files, **extra)
    train_set = format.read(train_select, numeric=True)
    pred = classify(files, label, align, train_set, **extra)
    format.write_dataset(output, align[:, 4], 1, label, pred)

def classify(files, label, align, train_idx, train_y, **extra):
    '''
    '''
    
    mask = create_mask(files, **extra)
    data = ndimage_processor.read_image_mat(files, label, image_transform, shared=False, mask=mask, cache_file=None, align=align, **extra)
    data -= data.min(axis=0)
    data /= data.max(axis=0)
    ha = 2*len(train_idx)/3
    print ha, len(train_idx)
    sys.stdout.flush()
    return SdA.classify(((data[train_idx[:ha].squeeze()], train_y[train_idx[:ha].squeeze()]), 
                         (data[train_idx[ha:].squeeze()], train_y[train_idx[ha:].squeeze()]), 
                         (data, train_y))
    )

def create_mask(files, pixel_diameter, resolution, apix, **extra):
    '''
    '''
    
    img = ndimage_file.read_image(files[0])
    mask = ndimage_utility.model_disk(int(pixel_diameter/2.0), img.shape)
    bin_factor = max(1, min(8, resolution / (apix*2))) if resolution > (2*apix) else 1
    _logger.info("Decimation factor %f for resolution %f and pixel size %f"%(bin_factor, resolution, apix))
    if bin_factor > 1: mask = ndimage_interpolate.downsample(mask, bin_factor)
    return mask

def image_transform(img, i, mask, resolution, apix, var_one=True, align=None, **extra):
    '''
    '''
    
    if align is not None:
        assert(align[i, 0]==0)
        if align[i, 1] > 179.999: img = ndimage_utility.mirror(img)
    ndimage_utility.vst(img, img)
    bin_factor = max(1, min(8, resolution / (apix*2))) if resolution > (2*apix) else 1
    if bin_factor > 1: img = ndimage_interpolate.downsample(img, bin_factor)
    ndimage_utility.normalize_standard(img, mask, var_one, img)
    if 1 == 0:
        img, freq = ndimage_utility.bispectrum(img, 0.5, 'gaussian', 1.0)
        freq;
        img = numpy.log10(numpy.sqrt(numpy.abs(img.real+1)))
    
    #img = ndimage_utility.compress_image(img, mask)
    return img

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
        align, header = format.read_alignment(alignment, spiderid=spiderid, ndarray=True)
        refidx = header.index('ref_num')
        label = numpy.zeros((len(align), 2), dtype=numpy.int)
        label[:, 0] = spider_utility.spider_id(files[0])
        label[:, 1] = align[:, 4].astype(numpy.int)-1
        if numpy.max(label[:, 1]) >= ndimage_file.count_images(files[0]):
            label[:, 1] = numpy.arange(0, len(align))
    else:
        align = None
        refidx = None
        if os.path.exists(alignment):
            _logger.debug("Alignment exists")
            align, header = format.read_alignment(alignment, ndarray=True)
            refidx = header.index('ref_num')
            if len(align)>0 and 'stack_id' in set(header):
                align = numpy.asarray(align)
                label = align[:, 15:17].astype(numpy.int)
                label[:, 1]-=1
            else:
                align=None
        if align is None:
            alignvals = []
            total = 0 
            for f in files:
                aligncur, header = format.read_alignment(alignment, spiderid=f, ndarray=True)
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
                    label[:, 1] = numpy.arange(0, len(align[total:end]))
                total = end
            align = numpy.asarray(alignvals)
    #ref = align[:, refidx].astype(numpy.int)
    #refs = numpy.unique(ref)
    return label, align


def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    #windows = ('none', 'uniform', 'sasaki', 'priestley', 'parzen', 'hamming', 'gaussian', 'daniell')
    group = OptionGroup(parser, "Classify", "Options to control particle classification",  id=__name__)

    group.add_option("", resolution=40.0,             help="Filter to given resolution - requires apix to be set")
    group.add_option("", use_rtsq=False,              help="Use alignment parameters to rotate projections in 2D")
    group.add_option("", train_select="",             help="Selection file containing labels for the training set")
    #group.add_option("", disable_bispec=False,        help="Use the image data rather than the bispectrum")
    #group.add_option("", bispec_window=windows,       help="Use the image data rather than the bispectrum", default=6)
    #group.add_option("", bispec_biased=False,         help="Estimate biased bispectrum, default unbiased")
    #group.add_option("", bispec_lag=0.314,            help="Percent of the signal to be used for lag")
    #group.add_option("", bispec_mode=('Both', 'Amp', 'Phase'), help="Part of the bispectrum to use", default=0)
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-a", alignment="",   help="Input file containing alignment parameters", required_file=True, gui=dict(filetype="open"))

def check_options(options, main_option=False):
    #Check if the option values are valid
    #from ..core.app.settings import OptionValueError
    
    #if options.bispec_lag < 1e-20: raise OptionValueError, "Invalid value for `--bispec-lag`, must be greater than zero"
    pass

def main():
    #Main entry point for this script
    program.run_hybrid_program(__name__,
        description = '''Clean a particle selection of any remaning bad windows
                        
                        http://
                        
                        Example:
                         
                        $ ara-autopart input-stack.spi -o coords.dat -p params.spi
                      ''',
        use_version = True,
        supports_OMP=True,
    )

def dependents(): return [spider_params]
if __name__ == "__main__": main()

