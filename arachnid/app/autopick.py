''' Automated particle selection (AutoPicker)

This script (`ara-autopick`) was designed locate particles on a micrograph using template matching 
(like LFCPick), yet incorporates several post-processing algorithms to reduce the number of noise 
windows and contaminants.

It will not remove all contaminants but experiments have demonstrated that in many cases it removes enough 
to achieve a high-resolution. To further reduce contamination, see ViCer (`ara-vicer`).

The AutoPicker script (`ara-autopick`) takes at minimum as input the micrograph and size of the particle in pixels and
writes out a set of coordinate files for each selected particle.

Tips
====

 #. Filenames: Must follow the SPIDER format with a number before the extension, e.g. mic_00001.spi. Output files just require the number of digits: `--output sndc_0000.spi`

 #. CCD micrographs - Use the `--invert` parameter to invert CCD micrographs. In this way, you can work directly with the original TIFF
 
 #. Decimation - Use the `--bin-factor` parameter to reduce the size of the micrograph for more efficient processing. Your coordinates will be on the full micrograph.
 
 #. Aggregration - Use `--remove-aggregates` to remove aggregation. This will remove all overlapping windows based on the window size controlled by `--window-mult`
    
 #. Parallel Processing - Several micrographs can be run in parallel (assuming you have the memory and cores available). `-p 8` will run 8 micrographs in parallel. 

Examples
========

.. sourcecode :: sh
    
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

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Filename for SPIDER parameter file describing a Cryo-EM experiment

Useful Options
===============

These options 

.. program:: ara-autopick

.. option:: -w <int>, --worker-count <int>
    
    Set the number of micrographs to process in parallel (keep in mind memory and processor restrictions)
    
.. option:: --invert
    
    Invert the contrast of CCD micrographs

.. option:: --bin-factor <FLOAT>
    
    Decimatation factor for the script: changes size of images, coordinates, parameters such as pixel_size or window unless otherwise specified

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

.. option:: --selection-file <str>
    
    Selection file for a subset of micrographs

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by MPI-enabled scripts... <mpi-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`

.. Created on Dec 21, 2011
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.image import ndimage_utility, ndimage_filter
from ..core.learn import dimensionality_reduction
from ..core.learn import unary_classification
from ..core.metadata import format_utility, format, spider_utility, spider_params
from ..core.parallel import mpi_utility
from ..core.util import drawing
import numpy # pylint: disable=W0611
import numpy.linalg
import scipy.spatial
import scipy.stats
import lfcpick
import logging, os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, id_len=0, **extra):
    '''Concatenate files and write to a single output file
        
    Args:
    
        filename : str
                   Input filename
        id_len : int, optional
                 Maximum length of the ID
        extra : dict
                Unused key word arguments
                
    Returns:
    
        filename : str
                   Current filename
        peaks : str
                Coordinates found
    '''
    
    try:
        spider_utility.update_spider_files(extra, spider_utility.spider_id(filename, id_len), 'good_coords', 'output', 'good', 'box_image')  
    except:
        _logger.info("Skipping: %s - invalid SPIDER ID"%filename)
        return filename, []
    _logger.debug("Read micrograph")
    mic = lfcpick.read_micrograph(filename, **extra)
    _logger.debug("Search micrograph")
    try:
        peaks = search(mic, **extra)
    except numpy.linalg.LinAlgError:
        _logger.info("Skipping: %s"%filename)
        return filename, []
    _logger.debug("Write coordinates")
    if len(peaks) == 0:
        _logger.info("Skipping: %s - no particles found"%filename)
        return filename, []
        
    coords = format_utility.create_namedtuple_list(peaks, "Coord", "id,peak,x,y",numpy.arange(1, len(peaks)+1, dtype=numpy.int)) if peaks.shape[0] > 0 else []
    write_example(mic, coords, filename, **extra)
    format.write(extra['output'], coords, default_format=format.spiderdoc)
    return filename, peaks

def search(img, disable_prune=False, limit=0, experimental=False, **extra):
    ''' Search a micrograph for particles using a template
    
    Args:
        
        img : array
              Micrograph image
        disable_prune : bool
                        Disable the removal of bad particles
        extra : dict
                Unused key word arguments
    
    Returns:
            
        peaks : array
                List of peaks: height and coordinates
    '''
    
    template = lfcpick.create_template(**extra)
    peaks = template_match(img, template, **extra)
    peaks=cull_boundary(peaks, img.shape, **extra)
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
            sel = classify_windows_experimental(img, peaks, **extra)
        else:
            sel = classify_windows(img, peaks, **extra)
        peaks = peaks[sel].copy()
    peaks[:, 1:3] *= extra['bin_factor']
    return peaks[::-1]

def template_match(img, template_image, pixel_diameter, **extra):
    ''' Find peaks using given template in the micrograph
    
    Args:
        
        img : array
              Micrograph
        template_image : array
                         Template image
        pixel_diameter : int
                         Diameter of particle in pixels
        extra : dict
                Unused key word arguments
          
    Returns:
        
        peaks : array
                List of peaks including peak size, x-coordinate, y-coordinate
    '''
    
    _logger.debug("Filter micrograph")
    img = ndimage_filter.gaussian_highpass(img, 0.25/(pixel_diameter/2.0), 2)
    _logger.debug("Template-matching")
    cc_map = ndimage_utility.cross_correlate(img, template_image)
    _logger.debug("Find peaks")
    peaks = lfcpick.search_peaks(cc_map, pixel_diameter, **extra)
    if peaks.ndim == 1: peaks = numpy.asarray(peaks).reshape((len(peaks)/3, 3))
    return peaks

def cull_boundary(peaks, shape, boundary=[], bin_factor=1.0, **extra):
    ''' Remove peaks where the window goes outside the boundary of the 
    micrograph image.
    
    Args:
        
        peaks : array
                List of peaks including peak size, x-coordinate, y-coordinate
        shape : tuple
                Number of rows, columns in micrograph
        boundary : list
                   Margin for particle selection top, bottom, left, right
        bin_factor : float
                     Image downsampling factor
        extra : dict
                Unused key word arguments
          
    Returns:
    
        peaks : array
                List of peaks within the boundary including peak size, x-coordinate, y-coordinate
    '''
    
    if len(boundary) == 0: return peaks
    boundary = numpy.asarray(boundary)/bin_factor
    
    if len(boundary) > 1: boundary[1] = shape[1]-boundary[1]
    if len(boundary) > 3: boundary[1] = shape[0]-boundary[3]
    
    _logger.debug("Boundary: %s"%(str(boundary)))
    j=0
    for i in xrange(len(peaks)):
        if peaks[i, 2] < boundary[0]: continue
        elif len(boundary) > 1 and peaks[i, 2] > boundary[1]: continue
        elif len(boundary) > 2 and peaks[i, 1] > boundary[2]: continue
        elif len(boundary) > 3 and peaks[i, 1] > boundary[3]: continue
        if i != j:
            peaks[j, :] = peaks[i]
        j+=1
    
    _logger.debug("Kept: %d of %d"%(j, len(peaks)))
    return peaks[:j]

def classify_windows(mic, scoords, dust_sigma=4.0, xray_sigma=4.0, disable_threshold=False, remove_aggregates=False, pca_mode=0, iter_threshold=1, real_space_nstd=2.5, window=None, pixel_diameter=None, threshold_minimum=25, **extra):
    ''' Classify particle windows from non-particle windows
    
    Args:
        
        mic : array
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
        real_space_nstd : float
                          Number of standard deviations for real-space PCA rejection
        window : int
                 Size of the window in pixels
        pixel_diameter : int
                         Diameter of particle in pixels
        threshold_minimum : int
                            Minimum number of consider success
        extra : dict
                Unused key word arguments
        
    Returns:

        sel : numpy.ndarray
              Bool array of selected good windows 
    '''
    
    _logger.debug("Total particles: %d"%len(scoords))
    radius = pixel_diameter/2
    win_shape = (window, window)
    dgmask = ndimage_utility.model_disk(radius/2, win_shape)
    masksm = dgmask
    maskap = ndimage_utility.model_disk(1, win_shape)*-1+1
    vfeat = numpy.zeros((len(scoords)))
    data = numpy.zeros((len(scoords), numpy.sum(masksm>0.5)))
    
    mask = ndimage_utility.model_disk(int(radius*1.2+1), (window, window)) * (ndimage_utility.model_disk(int(radius*0.9), win_shape)*-1+1)
    datar=None
    
    imgs=[]
    _logger.debug("Windowing %d particles"%len(scoords))
    for i, win in enumerate(ndimage_utility.for_each_window(mic, scoords, window, 1.0)):
        if (i%10)==0: _logger.debug("Windowing particle: %d"%i)
        #win=ndimage_filter.ramp(win)
        imgs.append(win.copy())
        
        ndimage_utility.replace_outlier(win, dust_sigma, xray_sigma, None, win)
        ar = ndimage_utility.compress_image(ndimage_utility.normalize_standard(win, mask, False), mask)
        
        if datar is None: datar=numpy.zeros((len(scoords), ar.shape[0])) 
        datar[i, :] = ar
        if vfeat is not None:
            vfeat[i] = numpy.sum(ndimage_utility.segment(ndimage_utility.dog(win, radius), 1024)*dgmask)
        amp=ndimage_utility.fftamp(win)*maskap
        ndimage_utility.vst(amp, amp)
        ndimage_utility.normalize_standard(amp, masksm, out=amp)
        ndimage_utility.compress_image(amp, masksm, data[i])
    
    _logger.debug("Performing PCA")
    feat, idx = dimensionality_reduction.pca(data, data, 1)[:2]
    if feat.ndim != 2:
        _logger.error("PCA bug: %s -- %s"%(str(feat.shape), str(data.shape)))
    assert(idx > 0)
    assert(feat.shape[0]>1)
    _logger.debug("Eigen: %d"%idx)
    dsel = unary_classification.one_class_classification_old(feat)
    

    feat, idx = dimensionality_reduction.pca(datar, datar, pca_mode)[:2]
    if feat.ndim != 2:
        _logger.error("PCA bug: %s -- %s"%(str(feat.shape), str(data.shape)))
    assert(idx > 0)
    assert(feat.shape[0]>1)
    _logger.debug("Eigen: %d"%idx)
    
    dsel = numpy.logical_and(dsel, unary_classification.robust_euclidean(feat, real_space_nstd))
    
    _logger.debug("Removed by PCA: %d of %d -- %d"%(numpy.sum(dsel), len(scoords), idx))
    if vfeat is not None:
        sel = numpy.logical_and(dsel, vfeat == numpy.max(vfeat))
        _logger.debug("Removed by Dog: %d of %d"%(numpy.sum(vfeat == numpy.max(vfeat)), len(scoords)))
    else: sel = dsel
    if not disable_threshold:
        for i in xrange(1, iter_threshold):
            tsel = classify_noise(scoords, dsel, sel, threshold_minimum)
            dsel = numpy.logical_and(dsel, numpy.logical_not(tsel))
            sel = numpy.logical_and(sel, numpy.logical_not(tsel))
        tsel = classify_noise(scoords, dsel, sel, threshold_minimum)
        _logger.debug("Removed by threshold %d of %d"%(numpy.sum(tsel), len(scoords)))
        sel = numpy.logical_and(tsel, sel)
        _logger.debug("Removed by all %d of %d"%(numpy.sum(sel), len(scoords)))
        sel = numpy.logical_and(dsel, sel)    
    
    if remove_aggregates: classify_aggregates(scoords, window/2, sel)
    #else: remove_overlap(scoords, radius, sel)
    return sel

def classify_windows_experimental(mic, scoords, dust_sigma=4.0, xray_sigma=4.0, disable_threshold=False, remove_aggregates=False, pca_mode=0, iter_threshold=1, real_space_nstd=2.5, window=None, pixel_diameter=None, threshold_minimum=25, **extra):
    ''' Classify particle windows from non-particle windows
    
    Args:
        
        mic : array
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
        real_space_nstd : float
                          Number of standard deviations for real-space PCA rejection
        window : int
                 Size of the window in pixels
        pixel_diameter : int
                         Diameter of particle in pixels
        threshold_minimum : int
                            Minimum number of consider success
        extra : dict
                Unused key word arguments
        
    Returns:

        sel : numpy.ndarray
              Bool array of selected good windows 
    '''
    
    _logger.debug("Total particles: %d"%len(scoords))
    radius = pixel_diameter/2
    win_shape = (window, window)
    dgmask = ndimage_utility.model_disk(radius/2, win_shape)
    masksm = dgmask
    maskap = ndimage_utility.model_disk(1, win_shape)*-1+1
    vfeat = numpy.zeros((len(scoords)))
    data = numpy.zeros((len(scoords), numpy.sum(masksm>0.5)))
    
    mask = ndimage_utility.model_disk(int(radius*1.2+1), (window, window)) * (ndimage_utility.model_disk(int(radius*0.9), win_shape)*-1+1)
    datar=None
    
    imgs=[]
    _logger.debug("Windowing %d particles"%len(scoords))
    for i, win in enumerate(ndimage_utility.for_each_window(mic, scoords, window, 1.0)):
        #if data is None:
        #    data = numpy.zeros((len(scoords), win.shape[0]/2-1))
        if (i%10)==0: _logger.debug("Windowing particle: %d"%i)
        #win=ndimage_filter.ramp(win)
        imgs.append(win.copy())
        
        ndimage_utility.replace_outlier(win, dust_sigma, xray_sigma, None, win)
        #ar = ndimage_utility.compress_image(ndimage_utility.normalize_standard(win, normmask, True), mask)
        ar = ndimage_utility.compress_image(ndimage_utility.normalize_standard(win, mask, False), mask)
        
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
    feat, idx = dimensionality_reduction.pca(data, data, 1)[:2]
    if feat.ndim != 2:
        _logger.error("PCA bug: %s -- %s"%(str(feat.shape), str(data.shape)))
    assert(idx > 0)
    assert(feat.shape[0]>1)
    _logger.debug("Eigen: %d"%idx)
    dsel = unary_classification.one_class_classification_old(feat)
    

    feat, idx = dimensionality_reduction.pca(datar, datar, pca_mode)[:2]
    if feat.ndim != 2:
        _logger.error("PCA bug: %s -- %s"%(str(feat.shape), str(data.shape)))
    assert(idx > 0)
    assert(feat.shape[0]>1)
    _logger.debug("Eigen: %d"%idx)
    
    dsel = numpy.logical_and(dsel, unary_classification.robust_euclidean(feat, real_space_nstd))
    
    _logger.debug("Removed by PCA: %d of %d -- %d"%(numpy.sum(dsel), len(scoords), idx))
    if vfeat is not None:
        sel = numpy.logical_and(dsel, vfeat == numpy.max(vfeat))
        _logger.debug("Removed by Dog: %d of %d"%(numpy.sum(vfeat == numpy.max(vfeat)), len(scoords)))
    else: sel = dsel
    if not disable_threshold:
        for i in xrange(1, iter_threshold):
            tsel = classify_noise(scoords, dsel, sel, threshold_minimum)
            dsel = numpy.logical_and(dsel, numpy.logical_not(tsel))
            sel = numpy.logical_and(sel, numpy.logical_not(tsel))
        tsel = classify_noise(scoords, dsel, sel, threshold_minimum)
        _logger.debug("Removed by threshold %d of %d"%(numpy.sum(tsel), len(scoords)))
        sel = numpy.logical_and(tsel, sel)
        _logger.debug("Removed by all %d of %d"%(numpy.sum(sel), len(scoords)))
        sel = numpy.logical_and(dsel, sel)    
    
    if remove_aggregates: classify_aggregates(scoords, window/2, sel)
    #else: remove_overlap(scoords, radius, sel)
    return sel

def outlier_rejection(feat, prob):
    '''
    '''
    
    from sklearn.covariance import EmpiricalCovariance #MinCovDet
    
    #real_cov
    #linalg.inv(real_cov)
    
    #robust_cov = MinCovDet().fit(feat)
    robust_cov = EmpiricalCovariance().fit(feat)
    dist = robust_cov.mahalanobis(feat - numpy.median(feat, 0))
    
    cut = scipy.stats.chi2.ppf(prob, feat.shape[1])
    return dist < cut
    
def classify_noise(scoords, dsel, sel=None, threshold_minimum=25):
    ''' Classify out the noise windows
    
    Args:
        
        scoords : list
                  List of peak and coordinates
        dsel : numpy.ndarray
               Good values selected by PCA
        sel : numpy.ndarray
               Total good values selected by PCA and DoG
        threshold_minimum : int
                            Minimum number of consider success
    
    Returns:
        
        tsel : numpy.ndarray
               Good values selected by Otsu
               
    '''
    
    if sel is None: sel = dsel
    bcnt = 0 
    tsel=None
    i=0
    while bcnt < threshold_minimum and i < 10:
        if tsel is not None: dsel = numpy.logical_and(numpy.logical_not(tsel), dsel)
        th = unary_classification.otsu(scoords[dsel, 0], numpy.sum(dsel)/16)
        tsel = scoords[:, 0] > th
        bcnt = numpy.sum(numpy.logical_and(dsel, numpy.logical_and(tsel, sel)))
        i+=1
    return tsel

def classify_aggregates(scoords, offset, sel):
    ''' Remove all aggregated windows
    
    Args:
        
        scoords : list
                  List of peak and coordinates
        offset : int
                 Window half-width
        sel : numpy.ndarray
              Good values selected by aggerate removal
    
    Returns:
            
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

def remove_overlap(scoords, radius, sel):
    ''' Remove coordinates where the windows overlap by updating
    the selection array (`sel`)
    
    Args:
     
        scoords : array
                  Selected coordinates
        radius : int
                 Radius of the particle in pixels 
        sel : array
              Output selection array that is modified in place
    
    '''
    
    coords = scoords[:, 1:3]
    i=0
    radius *= 1.1
    idx = numpy.argwhere(sel).squeeze()
    while i < len(idx):
        dist = scipy.spatial.distance.cdist(coords[idx[i+1:]], coords[idx[i]].reshape((1, len(coords[idx[i]]))), metric='euclidean').ravel()
        osel = dist < radius
        if numpy.sum(osel) > 0:
            if numpy.alltrue(scoords[idx[i], 0] > scoords[idx[i+1:], 0]):
                sel[idx[i+1:][osel]]=0
                idx = numpy.argwhere(sel).squeeze()
            else:
                sel[idx[i]]=0
        else:
            i+=1

def write_example(mic, coords, filename, box_image="", bin_factor=1.0, pixel_diameter=None, window=None, **extra):
    ''' Write out an image with the particles boxed
    
    Args:
    
        mic : array
              Micrograph image
        coords : list
                 List of particle coordinates
        filename : str
                   Current micrograph filename to load benchmark if available
        box_image : str
                    Output filename
        bin_factor : float
                     Image downsampling factor
        pixel_diameter : int
                         Diameter of particle in pixels
        window : int
                 Size of window in pixels
        extra : dict
                Unused key word arguments
    '''
    
    if box_image == "" or not drawing.is_available(): return
    
    radius = pixel_diameter/2.0
    mic = ndimage_filter.filter_gaussian_highpass(mic, 0.25/radius, 2)
    ndimage_utility.replace_outlier(mic, 4.0, 4.0, None, mic)
    
    bench = lfcpick.benchmark.read_bench_coordinates(filename, **extra)
    if bench is not None:
        mic = drawing.draw_particle_boxes(mic, coords, window, bin_factor, ret_draw=True)
        drawing.draw_particle_boxes_to_file(mic, bench, window, bin_factor, box_image, outline="#40ff40")
    else:
        drawing.draw_particle_boxes_to_file(mic, coords, window, bin_factor, box_image)
    

def initialize(files, param):
    # Initialize global parameters for the script
    
    if mpi_utility.is_root(**param):
        if param['disable_prune']: _logger.info("Bad particle removal - disabled")
        if param['disable_threshold']: _logger.info("Noise removal - disabled")
        if param['remove_aggregates']: _logger.info("Aggregate removal - enabled")
        if param['experimental']: _logger.info("Experimental contaminant removal - enabled")
        if len(param['boundary']) > 0: _logger.info("Selection boundary: %s"%",".join([str(v) for v in param['boundary']]))
        if param['iter_threshold']>1: _logger.info("Multiple-thresholds: %d"%param['iter_threshold'])
        if param['box_image']!="":
            try:os.makedirs(os.path.dirname(param['box_image']))
            except: pass
    return sorted(lfcpick.initialize(files, param))

def reduce_all(val, **extra):
    # Process each input file in the main thread (for multi-threaded code)

    return lfcpick.reduce_all(val, **extra)

def finalize(files, **extra):
    # Finalize global parameters for the script
    
    return lfcpick.finalize(files, **extra)

def supports(files, **extra):
    ''' Test if this module is required in the project workflow
    
    Args:
    
        files : list
                List of filenames to test
        extra : dict
                Unused keyword arguments
    
    Returns:
    
        flag : bool
               True if this module should be added to the workflow
    '''
    
    return True

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
    group.add_option("",   real_space_nstd=2.5,         help="Cutoff for real space PCA")
    group.add_option("",   boundary=[],                 help="Margin for particle selection top, bottom, left, right")
    group.add_option("",   threshold_minimum=25,        help="Minimum number of particles for threshold selection")
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", "--micrograph-files", input_files=[],     help="List of filenames for the input micrographs, e.g. mic_*.mrc", required_file=True, gui=dict(filetype="open"), regexp=spider_utility.spider_searchpath)
        pgroup.add_option("-o", "--coordinate-file",      output="",      help="Output filename for the coordinate file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("",   ctf_file="-",                             help="Input defocus file - currently ignored", required=True, gui=dict(filetype="open"))
        pgroup.add_option("-s", selection_file="",                        help="Selection file for a subset of good micrographs", gui=dict(filetype="open"), required_file=False)
        spider_params.setup_options(parser, pgroup, True)
        # move next three options to benchmark
        group = OptionGroup(parser, "Benchmarking", "Options to control benchmark particle selection",  id=__name__)
        group.add_option("-g", good="",        help="Good particles for performance benchmark", gui=dict(filetype="open"))
        group.add_option("",   good_coords="", help="Coordindates for the good particles for performance benchmark", gui=dict(filetype="open"))
        group.add_option("",   good_output="", help="Output coordindates for the good particles for performance benchmark", gui=dict(filetype="save"))
        group.add_option("",   box_image="",   help="Output filename for micrograph image with boxed particles - use `.png` as the extension", gui=dict(filetype="save"))
        pgroup.add_option_group(group)
        parser.change_default(log_level=3)

def change_option_defaults(parser):
    ''' Change the values to options specific to the script
    '''
    
    parser.change_default(bin_factor=4, window=1.35)
    
def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    if len(options.boundary) > 0:
        try: options.boundary = [int(v) for v in options.boundary]
        except: raise OptionValueError, "Unable to convert boundary margin to list of integers"

def flags():
    ''' Get flags the define the supported features
    
    Returns:
    
        flags : dict
                Supported features
    '''
    
    return dict(description = '''Automated particle selection (AutoPicker)
                        
                        $ ls input-stack_*.spi
                        input-stack_0001.spi input-stack_0002.spi input-stack_0003.spi
                        
                        Example: Unprocessed film micrograph
                         
                        $ ara-autopick input-stack_*.spi -o coords_00001.dat -r 110
                        
                        Example: Unprocessed CCD micrograph
                         
                        $ ara-autopick input-stack_*.spi -o coords_00001.dat -r 110 --invert
                      ''',
                supports_MPI=True, 
                supports_OMP=True,
                use_version=True)

def main():
    '''Main entry point for this script
    
    .. seealso:: 
    
        arachnid.core.app.program.run_hybrid_program
    
    '''
    program.run_hybrid_program(__name__)

def dependents():
    ''' List of depenent modules
    
    The autopick script depends on lfc for the template-matching 
    operations and uses many of the same parameters.
    
    .. seealso:: 
        
        arachnid.app.lfcpick
    
    Returns:
        
        modules : list
                  List of modules
    '''
    
    return [lfcpick]

if __name__ == "__main__": main()



