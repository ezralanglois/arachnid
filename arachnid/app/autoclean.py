'''
.. Created on Sep 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.image import ndimage_file, eman2_utility, analysis, ndimage_utility
from ..core.metadata import spider_utility, format, format_utility, spider_params
from ..core.parallel import mpi_utility
from ..core.image.ndplot import pylab
import logging, numpy, os, scipy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(input_vals, input_files, output, write_view_stack=0, sort_view_stack=False, **extra):
    '''Concatenate files and write to a single output file
        
    :Parameters:
    
    input_vals : list 
                 Tuple(view id, image labels and alignment parameters)
    input_files : list
                  List of input file stacks
    output : str
             Filename for output file
    write_view_stack : int
                       Write out views to a stack ('None', 'Positive', 'Negative', 'Both')
    sort_view_stack : bool
                      Write view stack sorted by first eigen vector
    extra : dict
            Unused key word arguments
                
    :Returns:
    
    filename : str
               Current filename
    peaks : str
            Coordinates found
    '''
    
    output = spider_utility.spider_filename(output, input_vals[0])
    data = read_data(input_files, *input_vals[1:], **extra)
    eigs, sel = classify_data(data, output=output, **extra)
    
    # Write 
    if eigs.ndim == 1: eigs = eigs.reshape((eigs.shape[0], 1))
    feat = numpy.hstack((sel[:, numpy.newaxis], eigs))
    header=[]
    for i in xrange(1, eigs.shape[1]+1): header.append('c%d'%i)
    format.write(os.path.splitext(output)[0]+".csv", feat, prefix="embed_", spiderid=input_vals[0], header=['select']+header)
    
    
    extra['poutput']=format_utility.new_filename(output, 'pos_') if write_view_stack == 1 or write_view_stack == 3 else None
    extra['noutput']=format_utility.new_filename(output, 'neg_') if write_view_stack == 2 or write_view_stack == 3 else None
    extra['sort_by']=eigs[:, 0] if sort_view_stack else None
    avg3 = compute_average3(input_files, *input_vals[1:], selected=sel, **extra)
    
    return input_vals, sel, avg3

def classify_data(data, output="", neig=1, **extra):
    ''' Classify the aligned projection data grouped by view
    
    :Parameters:
    
    data : array
           2D array where each row is an aligned and transformed image
    output : str
             Output filename for intermediary data
    neig : float
           Number of eigen vectors to use (or mode)
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    eigs : array
           2D array of embedded images
    sel : array
          1D array of selected images
    '''
    
    eigs = analysis.pca(data, frac=neig)[0] # Returns: val, idx, V[:idx], t[idx]
    
    # Sphere growing in Eigen Space
    cent = numpy.median(eigs, axis=0)
    eig_dist_cent = scipy.spatial.distance.cdist(eigs, cent.reshape((1, len(cent))), metric='euclidean').ravel()
    
    # Variance Estimate
    idx = numpy.argsort(eig_dist_cent)
    #var = analysis.running_variance(data[idx], axis=1)
    var = analysis.online_variance(data[idx], axis=0)
    rvar = analysis.online_variance(data[idx[::-1]], axis=0)
    #sel = numpy.argwhere(var[len(var)-1] < th).ravel()
    
    #slope = (var[len(var)-1, sel]-var[0, sel]) / len(var)
    #min_slope = numpy.max(slope)
    mvar = numpy.mean(var, axis=1)
    mrvar = numpy.mean(rvar, axis=1)
    
    if pylab is not None:
        maxval = numpy.max(var)
        pylab.clf()
        pylab.plot(numpy.arange(1, len(var)+1), var)
        pylab.savefig(format_utility.new_filename(output, "var_", ext="png"))
        pylab.clf()
        pylab.plot(numpy.arange(1, len(rvar)+1), rvar)
        pylab.savefig(format_utility.new_filename(output, "rvar_", ext="png"))
        pylab.clf()
        pylab.plot(numpy.arange(1, len(mrvar)+1), mrvar)
        pylab.savefig(format_utility.new_filename(output, "mrvar_", ext="png"))
        pylab.clf()
        pylab.plot(numpy.arange(1, len(var)+1), mvar, 'r-')
        pylab.savefig(format_utility.new_filename(output, "mvar_", ext="png"))
        
    if pylab is not None:
        th = analysis.otsu(eig_dist_cent)
        pylab.clf()
        n = pylab.hist(eig_dist_cent, bins=int(numpy.sqrt(len(mvar))))[0]
        maxval = sorted(n)[-1]
        pylab.plot((th, th), (0, maxval))
        pylab.savefig(format_utility.new_filename(output, "den_", ext="png"))
        _logger.info("Total selected: %d"%numpy.sum(eig_dist_cent<th))
    
    if 1 == 1:
        sel = eig_dist_cent < th
    elif 1 == 1:
        sel = numpy.zeros(len(eigs), dtype=numpy.bool)
        th = analysis.otsu(mvar)
        sel[idx[mvar<th]]=1
    else:
        sel = numpy.zeros(len(eigs), dtype=numpy.bool)
        sel[idx[:len(idx)/2]]=1
    
    return eigs, sel

def image_transform(img, align, mask, use_rtsq=False, resolution=0.0, apix=None, disable_bispec=False, bispec_window='gaussian', bispec_biased=False, bispec_lag=0.0081, bispec_srate=0.01, **extra):
    ''' Transform an image
    
    .. todo:: add ctf correction
    
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
    disable_bispec : bool
                     Disable the bispectrum representation
    bispec_window : str
                    Type of window to use for bispectrum
    bispec_biased : bool
                    Set true to estimate biased bispectrum
    bispec_lag : float
                 Percentage of the maximum lag
    bispec_srate : float
                   Bispectrum sample rate
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    out : array
          1D transformed image
    '''
    
    m = align[1] > 179.99
    if use_rtsq: img = eman2_utility.rot_shift2D(img, align[5], align[6], align[7], m)
    elif m:      img = eman2_utility.mirror(img)
    if resolution > (2*apix):
        img = eman2_utility.gaussian_low_pass(img, apix/resolution, True)
        bin_factor = max(1, min(8, resolution / (apix*2)))
    if bin_factor > 1: img = eman2_utility.decimate(img, bin_factor)
    if not disable_bispec:
        ndimage_utility.normalize_standard(img, None, True, img)
        scale = 'biased' if bispec_biased else 'unbiased'
        bispec_lag = int(bispec_lag*img.shape[0])
        bispec_lag = min(max(1, bispec_lag), img.ravel().shape[0])
        img *= mask
        try:
            img = ndimage_utility.bispectrum(img, bispec_srate, bispec_lag, bispec_window, scale)
        except:
            _logger.error("%d, %d"%(img.shape[0], bispec_lag))
            raise
        if 1 == 0: img = img.real
        elif 1 == 0: img = img.imag
        else:
            img = numpy.hstack((img.real[:, numpy.newaxis], img.imag[:, numpy.newaxis]))
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
            bin_factor = min(8, extra['resolution'] / (extra['apix']*2) ) if extra['resolution'] > (extra['apix']*2) else 1.0
            shape = numpy.asarray(img.shape) / bin_factor
            mask = ndimage_utility.model_disk(int(pixel_diameter/(2*bin_factor)), shape)
            _logger.info("Image size: (%d,%d) for bin-factor: %f, pixel size: %f, resolution: %f"%(mask.shape[0], mask.shape[1], bin_factor, extra['apix'], extra['resolution']))
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

def compute_average3(input_files, label, align, selected=None, use_rtsq=False, sort_by=None, noutput=None, poutput=None, **extra):
    ''' Compute the average of a selected, unselected and all images
    
    :Parameters:
    
    input_files : list
                  List of input file stacks
    label : array
            2D array of particle labels (stack_id, particle_id)
    align : array
            2D array of alignment parameters
    selected : array, optional
               Selected values to be included
    use_rtsq : bool
               Set true to rotate and translate
    output : str, optional
             Output filename for unseleced particles
    sort_by : bool
              Array to sort the output selected view stacks by
    noutput : str, optional
              Output filename for selected negative stack
    poutput : str, optional
              Output filename for selected positive stack
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    avgsel : array
             Averaged over selected images
    avgunsel : array
             Averaged over unselected images
    avgful : array
             Averaged over all images
    '''
    
    if sort_by is not None:
        idx = numpy.argsort(sort_by)
        label = label[idx]
        align = align[idx]
        if selected is not None:
            selected = selected[idx]
    
    avgsel, avgunsel = None, None
    cntsel  = numpy.sum(selected)
    cntunsel = selected.shape[0] - cntsel
    poff = 0
    noff = 0
    for i, img in enumerate(ndimage_file.iter_images(input_files, label)):
        m = align[i, 1] > 180.0
        if use_rtsq: img = eman2_utility.rot_shift2D(img, align[i, 5], align[i, 6], align[i, 7], m)
        elif m:      img = eman2_utility.mirror(img)
        if selected[i] > 0.5:
            if avgsel is None: avgsel = img.copy()
            else: avgsel += img
            if poutput is not None:
                ndimage_file.write_image(poutput, img, poff)
                poff += 1
        else:
            if avgunsel is None: avgunsel = img.copy()
            else: avgunsel += img
            if noutput is not None:
                ndimage_file.write_image(noutput, img, noff)
                noff += 1
    if avgunsel is None: avgunsel = numpy.zeros(img.shape)
    if avgsel is None: avgsel = numpy.zeros(img.shape)
    avgful = avgsel + avgunsel
    if cntsel > 0: avgsel /= cntsel
    if cntunsel > 0: avgunsel /= cntunsel
    return avgsel, avgunsel, avgful / (cntsel+cntunsel)

def initialize(files, param):
    # Initialize global parameters for the script
    
    if mpi_utility.is_root(**param):
        if param['resolution'] > 0.0: _logger.info("Filter and decimate to resolution: %f"%param['resolution'])
        if param['use_rtsq']: _logger.info("Rotate and translate data stack")
    
    group = None
    if mpi_utility.is_root(**param): 
        group = read_alignment(files, **param)
        _logger.info("Cleaning bad particles from %d views"%len(group))
    group = mpi_utility.broadcast(group, **param)
    if param['first_view']:
        tmp=group
        group = [tmp[0]]
    param['total'] = numpy.zeros((len(group), 3))
    param['sel_by_mic'] = {}
    
    return group

def reduce_all(val, input_files, total, sel_by_mic, output, file_completed, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    input, sel, avg3 = val
    label = input[1]
    file_completed -= 1
    total[file_completed, 0] = input[0]
    total[file_completed, 1] = numpy.sum(sel)
    total[file_completed, 2] = len(sel)
    for i in numpy.argwhere(sel):
        sel_by_mic.setdefault(int(label[i, 0]), []).append(int(label[i, 1]))    
    output = format_utility.add_prefix(output, "avg_")
    file_completed *= 3
    for i in xrange(len(avg3)):
        ndimage_file.write_image(output, avg3[i], file_completed+i)
    _logger.info("Finished processing %d - %d,%d"%(input[0], numpy.sum(total[:file_completed+1, 1]), numpy.sum(total[:file_completed+1, 2])))
    return input[0]

def finalize(files, total, sel_by_mic, output, **extra):
    # Finalize global parameters for the script
    
    format.write(output, total, prefix="sel_avg_", header=['id', 'selected', 'total'], default_format=format.spiderdoc)
    for id, sel in sel_by_mic.iteritems():
        sel = numpy.asarray(sel)
        format.write(output, numpy.vstack((sel, numpy.ones(sel.shape[0]))).T, prefix="sel_", spiderid=id, header=['id', 'select'], default_format=format.spidersel)
    _logger.info("Selected %d out of %d"%(numpy.sum(total[:, 1]), numpy.sum(total[:, 2])))
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    windows = ('none', 'uniform', 'sasaki', 'priestley', 'parzen', 'hamming', 'gaussian', 'daniell')
    view_stack = ('None', 'Positive', 'Negative', 'Both')
    group = OptionGroup(parser, "AutoPick", "Options to control reference-free particle selection",  id=__name__)
    group.add_option("", resolution=5.0,              help="Filter to given resolution - requires apix to be set")
    group.add_option("", use_rtsq=False,              help="Use alignment parameters to rotate projections in 2D")
    group.add_option("", neig=1.0,                    help="Number of eigen vectors to use")
    group.add_option("", disable_bispec=False,        help="Use the image data rather than the bispectrum")
    group.add_option("", bispec_window=windows,       help="Use the image data rather than the bispectrum", default=6)
    group.add_option("", bispec_biased=False,         help="Estimate biased bispectrum, default unbiased")
    group.add_option("", bispec_lag=0.314,            help="Percent of the signal to be used for lag")
    group.add_option("", bispec_srate=0.01,           help="Sample rate for the bispectrum")
    group.add_option("", first_view=False,            help="Test the algorithm on the first view")
    group.add_option("", write_view_stack=view_stack, help="Write out selected views to a stack", default=0)
    group.add_option("", sort_view_stack=False,       help="Sort the view stack by the first Eigen vector")
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-a", alignment="",   help="Input file containing alignment parameters", required_file=True, gui=dict(filetype="open"))
def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if options.bispec_lag < 1e-20: raise OptionValueError, "Invalid value for `--bispec-lag`, must be greater than zero"
    if options.bispec_srate < 1e-20: raise OptionValueError, "Invalid value for `--bispec-srate`, must be greater than zero"

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

def dependents(): return [spider_params]
if __name__ == "__main__": main()

