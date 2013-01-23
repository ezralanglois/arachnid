''' Automated particle selection (AutoPicker)

This script (`ara-autopick`) was designed locate particles on a micrograph using template matching 
(like LFCPick) but incorporates several post-processing algorithm to reduce the number of noise 
windows and contaminants.

It will not remove all contaminants but experiments have demonstrated that in many cases it removes enough 
to achieve a descent resolution. To remove more contaminants, use 2D classification (e.g. Relion). Currently,
I am working on a classification to further reduce contamination, this algorithm will be called AutoClean (`ara-autoclean`).

The AutoPicker script (`ara-autopick`) takes at minimum as input the micrograph and size of the particle in pixels and
writes out a set of coordinate files for each selected particle.

Tips
====

 #. Filenames: Must follow the SPIDER format with a number before the extension, e.g. mic_00001.spi. Output files just require the number of digits: `--output sndc_0000.spi`

 #. CCD micrographs - Use the `--invert` parameter to invert CCD micrographs. In this way, you can work directly with the original TIFF
 
 #. Decimation - Use the `--bin-factor` parameter to reduce the size of the micrograph for more efficient processing. Your coordinates will be on the full micrograph.
 
 #. Aggregration - Use `--remove-aggregates` to remove aggregation. This will remove all overlapping windows based on the window size controlled by `--window-mult`
 
 #. Restart - After a crash, you can restart where you left off by specifying restart file (a list of files already processed). One is automatically created in each run called
    .restart.autopick and can be used as follows: `--restart-file .restart.autopick`
    
 #. Parallel Processing - Several micrographs can be run in parallel (assuming you have the memory and cores available). `-p 8` will run 8 micrographs in parallel. 

Examples
========

.. sourcecode :: sh

    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Run with a disk as a template on a raw film micrograph
    
    $ ara-autopick mic_*.spi -o sndc_00001.spi -r 110 -w 312
    
    # Run with a disk as a template on a raw film micrograph on 16 cores (1 micrograph per core in memory)
    
    $ ara-autopick mic_*.spi -o sndc_00001.spi -r 110 -w 312 -p16
    
    # Run with a disk as a template on a raw CCD micrograph
    
    $ ara-autopick mic_*.spi -o sndc_00001.spi -r 110 -w 312 --invert
    
Critical Options
================

.. program:: ara-autopick

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of filenames for the input micrographs.
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`

.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for the coordinate file with correct number of digits (e.g. sndc_0000.spi)

.. option:: -r <int>, --pixel-radius <int>
    
    Size of your particle in pixels. If you decimate with `--bin-factor` give the undecimated pixel size.

Useful Options
===============

These options 

.. program:: ara-autopick

.. option:: -w <int>, --worker-count <int>
    
    Set the number of micrographs to process in parallel (keep in mind memory and processor restrictions)
    
.. option:: --invert
    
    Invert the contrast of CCD micrographs
    
.. option:: --bin-factor <int>
    
    Decimate the micrograph to speed up computation time
    
.. option:: --restart-file <FILENAME>

    If the script crashes, the restart file will allow it to pick up where it left off. If you did not specify one, 
     then .restart.autopick is automatically created. Just specify that as the filename on the next run and it will restart. If no
     restart file exists one is created with the name given (or .restart.autopick if none is given).

.. option:: --template <FILENAME>
    
    An input filename of a template to use in template-matching. If this is not specified then a Gaussian smoothed disk is used of radius 
    `disk-mult*pixel-radius`.

Tunable Options
===============

Generally, these options do not need to be changed, their default parameters have proven successful on many datasets. However,
you may enounter a dataset that does not react properly and these options can be adjusted to get the best possible particle
selection.

.. program:: ara-autopick

.. option:: -d <float>, --dust-sigma <float>
    
    Remove dark outlier pixels due to dust, suggested 3 for film 5 for CCD

.. option:: -x <float>, --xray-sigma <float>
    
    Remove light outlier pixels due to electrons, suggested 3 for film 5 for CCD

.. option:: --disable-prune
    
    Disable bad particle removal. This step is used to ensure the histogram has the proper bimodal distribution.

.. option:: --disable-threshold
    
    Disable noise removal. This step is used to remove the large number of noise windows.

.. option:: --dist-mult <float>
    
    This multipler scales the radius of the Gaussian smooth disk (which is used when no template is specified).

.. option:: --overlap-mult <float>
    
    Multiplier for the amount of allowed overlap or inter-particle distance.

.. option:: --pca-mode <float>

    Set the PCA mode for outlier removal: 0: auto, <1: energy, >=1: number of eigen vectors

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by MPI-enabled scripts... <mpi-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`

.. todo:: Test histogram

.. todo:: Test Version control

.. Created on Dec 21, 2011
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import tracing
from ..core.app.program import run_hybrid_program
from ..core.image import eman2_utility, ndimage_utility, analysis
from ..core.metadata import format_utility, format, spider_utility
from ..core.parallel import mpi_utility
import numpy, scipy, logging, scipy.misc
import lfcpick

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from PIL import ImageDraw 
    ImageDraw;
except: 
    try:
        import ImageDraw
        ImageDraw;
    except: 
        tracing.log_import_error('Failed to load PIL - `--box-image` will not work', _logger)
        ImageDraw = None


def process(filename, id_len=0, confusion=[], **extra):
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
    
    spider_utility.update_spider_files(extra, spider_utility.spider_id(filename, id_len), 'good_coords', 'output', 'good', 'box_image')  
    _logger.debug("Read micrograph")
    mic = lfcpick.read_micrograph(filename, **extra)
    _logger.debug("Search micrograph")
    peaks = search(mic, **extra)
    _logger.debug("Write coordinates")
    coords = format_utility.create_namedtuple_list(peaks, "Coord", "id,peak,x,y", numpy.arange(1, peaks.shape[0]+1, dtype=numpy.int))
    write_example(mic, coords, **extra)
    format.write(extra['output'], coords, default_format=format.spiderdoc)
    return filename, peaks

def write_example(mic, coords, box_image="", **extra):
    ''' Write out an image with the particles boxed
    
    :Parameters:
    
    coords : list
             List of particle coordinates
    box_image : str
                Output filename
    '''
    
    if box_image == "" or ImageDraw is None: return
    
    width, bin_factor = init_param(**param)[1:3]
    mic = scipy.misc.toimage(mic).convert("RGB")
    draw = ImageDraw.Draw(mic)
    
    width=int((window_size / float(bin_factor))*0.5)
    for box in coords:
        x = box.x / bin_factor
        y = box.y / bin_factor
        draw.rectangle((x+width, y+width, x-width, y-width), fill=None, outline="#ff4040")
    mic.save(output_file)

def search(img, overlap_mult=1.2, disable_prune=False, limit=0, experimental=False, **extra):
    ''' Search a micrograph for particles using a template
    
    :Parameters:
        
    img : EMData
          Micrograph image
    overlap_mult : float
                   Amount of allowed overlap
    disable_prune : bool
                    Disable the removal of bad particles
    extra : dict
            Unused key word arguments
    
    :Returns:
        
    peaks : numpy.ndarray
            List of peaks and coordinates
    '''
    
    template = lfcpick.create_template(**extra)
    radius, offset, bin_factor, mask = lfcpick.init_param(**extra)
    _logger.debug("Filter micrograph")
    img = eman2_utility.gaussian_high_pass(img, 0.25/radius, True)
    _logger.debug("Template-matching")
    cc_map = ccf_center(img, template)
    _logger.debug("Find peaks")
    peaks = lfcpick.search_peaks(cc_map, radius, overlap_mult)
    if peaks.ndim == 1: peaks = numpy.asarray(peaks).reshape((len(peaks)/3, 3))
    index = numpy.argsort(peaks[:,0])[::-1]
    if index.shape[0] > limit: index = index[:limit]
    index = index[::-1]
    try:
        peaks = peaks[index].copy().squeeze()
    except:
        _logger.error("%d > %d"%(numpy.max(index), peaks.shape[0]))
        raise
    if not disable_prune:
        _logger.debug("Classify peaks")
        if experimental:
            sel = classify_windows_new(img, peaks, **extra)
        else:
            sel = classify_windows(img, peaks, **extra)
        peaks = peaks[sel].copy()
    peaks[:, 1:3] *= bin_factor
    return peaks[::-1]

def ccf_center(img, template):
    ''' Noise-cancelling cross-correlation
    
    :Parameters:
        
    img : EMData
          Micrograph
    template : EMData
          Template
    
    :Returns:
        
    cc_map : EMData
             Cross-correlation map
    '''
    
    if eman2_utility.is_em(img):
        emimg = img
        img = eman2_utility.em2numpy(emimg)
    if eman2_utility.is_em(template):
        emtemplate = template
        template = eman2_utility.em2numpy(emtemplate)
    cc_map = ndimage_utility.cross_correlate(img, template)
    #cc_map = eman2_utility.numpy2em(cc_map)
    return cc_map

def classify_windows(mic, scoords, dust_sigma=4.0, xray_sigma=4.0, disable_threshold=False, remove_aggregates=False, pca_real=False, pca_mode=0, iter_threshold=1, **extra):
    ''' Classify particle windows from non-particle windows
    
    :Parameters:
        
    mic : EMData
          Micrograph
    scoords : list
              List of potential particle coordinates
    dust_sigma : float
                 Number of standard deviations for removal of outlier dark pixels
    xray_sigma : float
                 Number of standard deviations for removal of outlier light pixels
    disable_threshold : bool
                        Disable noise removal with threshold selection
    remove_aggregates : bool
                        Set True to remove aggregates
    pca_mode : float
               Set the PCA mode for outlier removal: 0: auto, <1: energy, >=1: number of eigen vectors
    iter_threshold : int
                     Number of times to repeat thresholding
    extra : dict
            Unused key word arguments
    
    :Returns:
        
    sel : numpy.ndarray
          Bool array of selected good windows 
    '''
    
    _logger.debug("Total particles: %d"%len(scoords))
    radius, offset, bin_factor, mask = lfcpick.init_param(**extra)
    emdata = eman2_utility.utilities.model_blank(offset*2, offset*2)
    npdata = eman2_utility.em2numpy(emdata)
    dgmask = ndimage_utility.model_disk(radius/2, offset*2)
    masksm = dgmask
    maskap = ndimage_utility.model_disk(1, offset*2)*-1+1
    vfeat = None #numpy.zeros((len(scoords)))
    data = numpy.zeros((len(scoords), numpy.sum(masksm>0.5)))
    if eman2_utility.is_em(mic):
        emmic = mic
        mic = eman2_utility.em2numpy(emmic)

    bin_factor=1.0
    _logger.debug("Windowing %d particles"%len(scoords))
    for i, win in enumerate(ndimage_utility.for_each_window(mic, scoords, offset*2, bin_factor)):
        if (i%10)==0: _logger.debug("Windowing particle: %d"%i)
        if 1 == 0:
            npdata[:, :] = win
            eman2_utility.ramp(emdata)
            win[:, :] = npdata
        ndimage_utility.replace_outlier(win, dust_sigma, xray_sigma, None, win)
        if vfeat is not None:
            vfeat[i] = numpy.sum(ndimage_utility.segment(ndimage_utility.dog(win, radius), 1024)*dgmask)
        amp = ndimage_utility.fourier_mellin(win)*maskap
        ndimage_utility.vst(amp, amp)
        ndimage_utility.normalize_standard(amp, masksm, out=amp)
        ndimage_utility.compress_image(amp, masksm, data[i])
    
    _logger.debug("Performing PCA")
    feat, idx = analysis.pca(data, data, pca_mode)[:2]
    if feat.ndim != 2:
        _logger.error("PCA bug: %s -- %s"%(str(feat.shape), str(data.shape)))
    assert(idx > 0)
    assert(feat.shape[0]>1)
    _logger.debug("Eigen: %d"%idx)
    dsel = analysis.one_class_classification_old(feat)
    
    _logger.debug("Removed by PCA: %d of %d -- %d"%(numpy.sum(dsel), len(scoords), idx))
    if vfeat is not None:
        sel = numpy.logical_and(dsel, vfeat == numpy.max(vfeat))
        _logger.debug("Removed by Dog: %d of %d"%(numpy.sum(vfeat == numpy.max(vfeat)), len(scoords)))
    else: sel = dsel
    if not disable_threshold:
        for i in xrange(1, iter_threshold):
            tsel = classify_noise(scoords, dsel, sel)
            dsel = numpy.logical_and(dsel, numpy.logical_not(tsel))
            sel = numpy.logical_and(sel, numpy.logical_not(tsel))
        tsel = classify_noise(scoords, dsel, sel)
        _logger.debug("Removed by threshold %d of %d"%(numpy.sum(tsel), len(scoords)))
        sel = numpy.logical_and(tsel, sel)
        _logger.debug("Removed by all %d of %d"%(numpy.sum(sel), len(scoords)))      

    if remove_aggregates: classify_aggregates(scoords, offset, sel)
    return sel

def classify_windows_new(mic, scoords, dust_sigma=4.0, xray_sigma=4.0, disable_threshold=False, remove_aggregates=False, pca_real=False, pca_mode=0, iter_threshold=1, **extra):
    ''' Classify particle windows from non-particle windows
    
    :Parameters:
        
    mic : EMData
          Micrograph
    scoords : list
              List of potential particle coordinates
    dust_sigma : float
                 Number of standard deviations for removal of outlier dark pixels
    xray_sigma : float
                 Number of standard deviations for removal of outlier light pixels
    disable_threshold : bool
                        Disable noise removal with threshold selection
    remove_aggregates : bool
                        Set True to remove aggregates
    pca_mode : float
               Set the PCA mode for outlier removal: 0: auto, <1: energy, >=1: number of eigen vectors
    iter_threshold : int
                     Number of times to repeat thresholding
    extra : dict
            Unused key word arguments
    
    :Returns:
        
    sel : numpy.ndarray
          Bool array of selected good windows 
    '''
    
    _logger.debug("Total particles: %d"%len(scoords))
    radius, offset, bin_factor, mask = lfcpick.init_param(**extra)
    #emdata = eman2_utility.utilities.model_blank(offset*2, offset*2)
    #npdata = eman2_utility.em2numpy(emdata)
    dgmask = ndimage_utility.model_disk(radius/2, offset*2)
    masksm = dgmask
    maskap = ndimage_utility.model_disk(1, offset*2)*-1+1
    vfeat = numpy.zeros((len(scoords)))
    data = numpy.zeros((len(scoords), numpy.sum(masksm>0.5)))
    #data = numpy.zeros((len(scoords), masksm.ravel().shape[0]))
    #data = None #numpy.zeros((len(scoords), numpy.sum(masksm>0.5)))
    if eman2_utility.is_em(mic):
        emmic = mic
        mic = eman2_utility.em2numpy(emmic)
    
    npmask = eman2_utility.em2numpy(mask)
    npmask[:] = eman2_utility.model_circle(int(radius*1.2+1), offset*2, offset*2) * (eman2_utility.model_circle(int(radius*0.9), offset*2, offset*2)*-1+1)
    datar=None
    
    imgs=[]
    bin_factor=1.0
    _logger.debug("Windowing %d particles"%len(scoords))
    for i, win in enumerate(ndimage_utility.for_each_window(mic, scoords, offset*2, bin_factor)):
        #if data is None:
        #    data = numpy.zeros((len(scoords), win.shape[0]/2-1))
        if (i%10)==0: _logger.debug("Windowing particle: %d"%i)
        #npdata[:, :] = win
        #eman2_utility.ramp(emdata)
        #win[:, :] = npdata
        ndimage_utility.ramp(win,win)
        
        ndimage_utility.replace_outlier(win, dust_sigma, xray_sigma, None, win)
        #ar = ndimage_utility.compress_image(ndimage_utility.normalize_standard(win, normmask, True), npmask)
        ar = ndimage_utility.compress_image(ndimage_utility.normalize_standard(win, npmask, False), npmask)
        
        imgs.append(win.copy())
        if datar is None: datar=numpy.zeros((len(scoords), ar.shape[0])) 
        datar[i, :] = ar
        if vfeat is not None:
            vfeat[i] = numpy.sum(ndimage_utility.segment(ndimage_utility.dog(win, radius), 1024)*dgmask)
        #amp = ndimage_utility.fourier_mellin(win)*maskap
        amp=ndimage_utility.fftamp(win)*maskap
        #amp = ndimage_utility.powerspec1d(win)
        ndimage_utility.vst(amp, amp)
        ndimage_utility.normalize_standard(amp, masksm, out=amp)
        if 1 == 1:
            ndimage_utility.compress_image(amp, masksm, data[i])
        else:
            data[i, :]=amp
    
    _logger.debug("Performing PCA")
    feat, idx = analysis.pca(data, data, 1)[:2]
    if feat.ndim != 2:
        _logger.error("PCA bug: %s -- %s"%(str(feat.shape), str(data.shape)))
    assert(idx > 0)
    assert(feat.shape[0]>1)
    _logger.debug("Eigen: %d"%idx)
    dsel = analysis.one_class_classification_old(feat)
    
    feat, idx = analysis.pca(datar, datar, pca_mode)[:2]
    if feat.ndim != 2:
        _logger.error("PCA bug: %s -- %s"%(str(feat.shape), str(data.shape)))
    assert(idx > 0)
    assert(feat.shape[0]>1)
    _logger.debug("Eigen: %d"%idx)
    
    for i in xrange(feat.shape[1]):
        if dsel is None:
            dsel = analysis.robust_rejection(numpy.abs(feat[:, i]), 2.5)
        else:
            dsel = numpy.logical_and(dsel, analysis.robust_rejection(numpy.abs(feat[:, i]), 2.5))
    dsel = numpy.logical_and(dsel, vfeat == numpy.max(vfeat))
        
    
    if 1 == 0:
        import pylab
        from arachnid.core.util import plotting
        pylab.clf()
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        nsel = numpy.logical_not(dsel)
        ax.scatter(scoords[dsel, 0], feat[dsel, 0])
        ax.scatter(scoords[nsel, 0], feat[nsel, 0], color='r')
        image_size=0.9
        radius=40
        index = plotting.nonoverlapping_subset(ax, scoords[:, 0], feat[:, 0], radius, 100)
        imgs2 = [imgs[i] for i in index]
        plotting.plot_images(fig, imgs2, scoords[index, 0], feat[index, 0], image_size, radius)
        pylab.savefig("scatter_cc_rpca.png")
    
    _logger.debug("Removed by PCA: %d of %d -- %d"%(numpy.sum(dsel), len(scoords), idx))
    if vfeat is not None:
        sel = numpy.logical_and(dsel, vfeat == numpy.max(vfeat))
        _logger.debug("Removed by Dog: %d of %d"%(numpy.sum(vfeat == numpy.max(vfeat)), len(scoords)))
    else: sel = dsel
    if not disable_threshold:
        for i in xrange(1, iter_threshold):
            tsel = classify_noise(scoords, dsel, sel)
            dsel = numpy.logical_and(dsel, numpy.logical_not(tsel))
            sel = numpy.logical_and(sel, numpy.logical_not(tsel))
        tsel = classify_noise(scoords, dsel, sel)
        _logger.debug("Removed by threshold %d of %d"%(numpy.sum(tsel), len(scoords)))
        sel = numpy.logical_and(tsel, sel)
        _logger.debug("Removed by all %d of %d"%(numpy.sum(sel), len(scoords)))
        sel = numpy.logical_and(dsel, sel)
    if remove_aggregates: classify_aggregates(scoords, offset, sel)
    return sel
    
def classify_noise(scoords, dsel, sel=None):
    ''' Classify out the noise windows
    
    :Parameters:
        
    scoords : list
              List of peak and coordinates
    dsel : numpy.ndarray
           Good values selected by PCA
    sel : numpy.ndarray
           Total good values selected by PCA and DoG
    
    :Returns:
        
    tsel : numpy.ndarray
           Good values selected by Otsu
               
    '''
    
    if sel is None: sel = dsel
    bcnt = 0 
    tsel=None
    while bcnt < 25:
        if tsel is not None: dsel = numpy.logical_and(numpy.logical_not(tsel), dsel)
        th = analysis.otsu(scoords[dsel, 0], numpy.sum(dsel)/16)
        tsel = scoords[:, 0] > th
        bcnt = numpy.sum(numpy.logical_and(dsel, numpy.logical_and(tsel, sel)))
    return tsel

def classify_aggregates(scoords, offset, sel):
    ''' Classify out the aggregate windows
    
    :Parameters:
        
    scoords : list
              List of peak and coordinates
    offset : int
             Window half-width
    sel : numpy.ndarray
          Good values selected by aggerate removal
    
    :Returns:
        
    sel : numpy.ndarray
          Good values selected by aggregate removal
               
    '''
    
    cutoff = offset*2
    coords = scoords[sel, 1:3]
    off = numpy.argwhere(sel).squeeze()
    dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords, 'euclidean'))
    dist = numpy.unique(numpy.argwhere(numpy.logical_and(dist > 0, dist <= cutoff)).ravel())
    sel[off[dist]] = 0
    return sel

def initialize(files, param):
    # Initialize global parameters for the script
    
    if mpi_utility.is_root(**param):
        if param['disable_prune']: _logger.info("Bad particle removal - disabled")
        if param['disable_threshold']: _logger.info("Noise removal - disabled")
        if param['remove_aggregates']: _logger.info("Aggregate removal - enabled")
    return lfcpick.initialize(files, param)

def reduce_all(val, **extra):
    # Process each input file in the main thread (for multi-threaded code)

    return lfcpick.reduce_all(val, **extra)

def finalize(files, **extra):
    # Finalize global parameters for the script
    
    return lfcpick.finalize(files, **extra)

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "AutoPick", "Options to control reference-free particle selection",  id=__name__)
    group.add_option("-d", dust_sigma=4.0,              help="Remove dark outlier pixels due to dust, suggested 3 for film 5 for CCD", gui=dict(maximum=100, minimum=0, singleStep=1))
    group.add_option("-x", xray_sigma=4.0,              help="Remove light outlier pixels due to electrons, suggested 3 for film 5 for CCD", gui=dict(maximum=100, minimum=1, singleStep=1))
    group.add_option("",   disable_prune=False,         help="Disable bad particle removal")
    group.add_option("",   disable_threshold=False,     help="Disable noise thresholding")
    group.add_option("",   remove_aggregates=False,     help="Use difference of Gaussian to remove possible aggergates (only use this option if there are many)")
    group.add_option("",   pca_mode=1.0,                help="Set the PCA mode for outlier removal: 0: auto, <1: energy, >=1: number of eigen vectors", gui=dict(minimum=0.0))
    group.add_option("",   iter_threshold=1,            help="Number of times to iterate thresholding")
    group.add_option("",   limit=2000,                  help="Limit on number of particles, 0 means give all", gui=dict(minimum=0, singleStep=1))
    group.add_option("",   experimental=False,          help="Use the latest experimental features!")
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        # move next three options to benchmark
        group = OptionGroup(parser, "Benchmarking", "Options to control benchmark particle selection",  id=__name__)
        group.add_option("-g", good="",        help="Good particles for performance benchmark", gui=dict(filetype="open"))
        group.add_option("",   good_coords="", help="Coordindates for the good particles for performance benchmark", gui=dict(filetype="open"))
        group.add_option("",   good_output="", help="Output coordindates for the good particles for performance benchmark", gui=dict(filetype="open"))
        group.add_option("",   box_image="",   help="Output filename for micrograph image with boxed particles - use `.png` as the extension")
        pgroup.add_option_group(group)
        parser.change_default(log_level=3)

def setup_main_options(parser, group):
    ''' 
    '''
    
    parser.change_default(bin_factor=4, window=1.35)
    

def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    if options.pixel_radius == 0: raise OptionValueError, "Pixel radius must be greater than zero"

def main():
    #Main entry point for this script
    run_hybrid_program(__name__,
        description = '''Find particles using template-matching with unsupervsied learning algorithm
                        
                        http://
                        
                        Example: Unprocessed film micrograph
                         
                        $ ara-autopick input-stack.spi -o coords.dat -r 110
                        
                        Example: Unprocessed CCD micrograph
                         
                        $ ara-autopick input-stack.spi -o coords.dat -r 110 --invert
                      ''',
        use_version = True,
        supports_OMP=True,
    )

def dependents(): return [lfcpick]
if __name__ == "__main__": main()



