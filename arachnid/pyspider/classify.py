''' Classify projections based on an alignment

This |spi| batch file (`spi-classify`) analyzes an alignment file and selects good projections based
on the specified parameters.

Tips
===

 #.  When performing cross-correlation cutoffs by view, make sure you understand how the :option:`view-resolution`
     works. It essentially coarse grains the alignment grid to ensure a reasonable number of particles for automated
     thresholding.

Examples
========

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Reconstruct a volume using a raw data stack
    
    # Automatically threshold each view where resolution 2 means 104 views
    
    $ spi-classify align1.spi align2.spi -o select01.spi --threshold-type Auto --threshold-by-view --view-resolution 2

Critical Options
================

.. program:: spi-classify

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames containing alignment parameters.
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`
    
.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for the selection file

.. option:: --threshold-type <(None, Auto, CC, Total)>

    Type of thresholding to perform
    
Useful Options
==============
    
.. option:: --cc-threshold <FLOAT>

    Cross-correlation threshold value (used with `--threshold-type CC`)
    
.. option:: --cc-total <FLOAT>

    Total number of high cross-correlation projections to keep, if < 1.0, assumes a fraction, otherwise total number (used with `--threshold-type Total`)
    
.. option:: --threshold-by-view <BOOL>

    Apply threshold within each view (determined with `view_resolution`)
    
.. option:: --view-resolution <INT>

    Resolution (or angular increment) to consider, e.g. 1 ~ 30 deg (theta-delta), 2 ~ 15 deg, 3 ~ 7 deg (used when `--threshold-by-view` is set True)
    
.. option:: --cull-overrep <BOOL>

    Set to True if you want to ensure each view has roughly the same number of particles (used when `--threshold-by-view` is set True)
    
.. option:: --cc-nstd <INT>

    Remove low cross-correlation particles whose cross-correlation score is less than cc_nstd times the standard deviation outside the mean

Advanced Options
================

.. option:: --threshold-bins <INT>

    Number of bins to use in Otzu's method: 0 = SQRT(total), -1 = total/16, otherwise use the number given (used with `--threshold-type Auto`)
    
.. option:: --keep-low-cc <BOOL>

    Set to True if you want to keep the low cross-correlation particles instead


Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`

.. todo:: add classify by defocus group to classify by view, and alone

.. Created on Aug 15, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.metadata import format, spider_utility
from ..core.image import analysis
from ..core.orient import healpix
import logging, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, **extra):
    ''' Create a reference from from a given density map
    
    :Parameters:
    
    filename : str
               Input alignment file
    output : str
             Output selection file
    extra : dict
            Unused key word arguments
             
    :Returns:
    
    filename : str
               Filename for correct location
    '''
    
    if spider_utility.is_spider_filename(filename):
        output = spider_utility.spider_filename(output, filename)
    align = numpy.asarray(format.read(filename, numeric=True))
    sel = classify_projections(align, **extra)
    if sel is not None:
        format.write(output, numpy.argwhere(sel)+1, header=['id'])
    return filename

def classify_projections(alignvals, threshold_type=('None', 'Auto', 'CC', 'Total'), cc_threshold=0.0, cc_total=0.9, threshold_by_view=False, view_resolution=1, threshold_bins=0, keep_low_cc=False, cull_overrep=False, cc_nstd=3, **extra):
    '''Classify projections based on alignment parameters
    
    :Parameters:
    
    alignvals : numpy.ndarray
                Alignment values
    threshold_type : int
                     Type of thresholding to perform
    cc_threshold : float
                   Cross-correlation threshold value (used with `--threshold-type CC`)
    cc_total : float
               Total number of high cross-correlation projections to keep, if < 1.0, assumes a fraction, otherwise total number (used with `--threshold-type Total`)
    threshold_by_view : bool
                        Apply threshold within each view (determined with `view_resolution`)
    view_resolution : int
                      Resolution (or angular increment) to consider, e.g. 1 ~ 30 deg (theta-delta), 2 ~ 15 deg, 3 ~ 7 deg (used when `--threshold-by-view` is set True)
    threshold_bins : int
                     Number of bins to use in Otzu`s method: 0 = SQRT(total), -1 = total/16, otherwise use the number given (used with `--threshold-type Auto`)
    keep_low_cc : bool
                  Set to True if you want to keep the low cross-correlation particles instead
    cull_overrep : bool
                   Set to True if you want to ensure each view has roughly the same number of particles (used when `--threshold-by-view` is set True)
    cc_nstd : int
              Remove low cross-correlation particles whose cross-correlation score is less than cc_nstd times the standard deviation outside the mean
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    selected : numpy.ndarray
               Boolean array of selected projections
    
    '''
    
    if threshold_type == 0: return None
    if threshold_by_view:
        return classify_projections_by_view(alignvals, threshold_type, cc_threshold, cc_total, view_resolution, threshold_bins, keep_low_cc, cull_overrep, cc_nstd)
    else:
        return classify_projections_overall(alignvals, threshold_type, cc_threshold, cc_total, threshold_bins, keep_low_cc, cc_nstd)

def classify_projections_overall(alignvals, threshold_type=('None', 'Auto', 'CC', 'Total'), cc_threshold=0.0, cc_total=0.9, threshold_bins=0, keep_low_cc=False, cc_nstd=3, **extra):
    ''' Classify projections based on alignment parameters by view
    
    :Parameters:
    
    alignvals : numpy.ndarray
                Alignment values
    threshold_type : int
                     Type of thresholding to perform
    cc_threshold : float
                   Cross-correlation threshold value (used with `--threshold-type CC`)
    cc_total: float
              Total number of high cross-correlation projections to keep, if < 1.0, assumes a fraction, otherwise total number (used with `--threshold-type Total`)
    threshold_bins : int
                     Number of bins to use in Otzu's method: 0 = SQRT(total), -1 = total/16, otherwise use the number given (used with `--threshold-type Auto`)
    keep_low_cc : bool
                  Set to True if you want to keep the low cross-correlation particles instead
    cc_nstd : int
              Remove low cross-correlation particles whose cross-correlation score is less than cc_nstd times the standard deviation outside the mean
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    selected : numpy.ndarray
               Boolean array of selected projections
    '''
    
    
    if threshold_type == 0: return None
    sel = analysis.one_class_selection(alignvals[:, 10], cc_nstd) if cc_nstd > 0 else numpy.ones(alignvals.shape[0], dtype=numpy.bool)
    cmp = numpy.less if keep_low_cc else numpy.greater
    cc = alignvals[sel, 10]
    if threshold_type == 1:
        cc_threshold = analysis.otsu(cc, threshold_bins)
    elif threshold_type == 3:
        cc_threshold = analysis.threshold_from_total(cc, cc_total, keep_low_cc)
    sel = numpy.logical_and(sel, cmp(alignvals[:, 10], cc_threshold))
    _logger.info("Overall Threshold: %f -> %d of %d"%(cc_threshold, numpy.sum(sel), len(alignvals)))
    return sel

def classify_projections_by_view(alignvals, threshold_type=('None', 'Auto', 'CC', 'Total'), cc_threshold=0.0, cc_total=0.9, view_resolution=1, threshold_bins=0, keep_low_cc=False, cull_overrep=False, cc_nstd=3, **extra):
    ''' Classify projections based on alignment parameters by view
    
    :Parameters:
    
    alignvals : numpy.ndarray
                Alignment values
    threshold_type : int
                     Type of thresholding to perform
    cc_threshold : float
                   Cross-correlation threshold value (used with `--threshold-type CC`)
    cc_total: float
              Total number of high cross-correlation projections to keep, if < 1.0, assumes a fraction, otherwise total number (used with `--threshold-type Total`)
    view_resolution : int
                      Resolution (or angular increment) to consider, e.g. 1 ~ 30 deg (theta-delta), 2 ~ 15 deg, 3 ~ 7 deg (used when `--threshold-by-view` is set True)
    threshold_bins : int
                     Number of bins to use in Otzu's method: 0 = SQRT(total), -1 = total/16, otherwise use the number given (used with `--threshold-type Auto`)
    keep_low_cc : bool
                  Set to True if you want to keep the low cross-correlation particles instead
    cull_overrep : bool
                   Set to True if you want to ensure each view has roughly the same number of particles (used when `--threshold-by-view` is set True)
    cc_nstd : int
              Remove low cross-correlation particles whose cross-correlation score is less than cc_nstd times the standard deviation outside the mean
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    selected : numpy.ndarray
               Boolean array of selected projections
    '''
      
    
    if threshold_type == 0: return None
    cmp = numpy.less if keep_low_cc else numpy.greater
    sel = analysis.robust_rejection(alignvals[:, 10], cc_nstd) if cc_nstd > 0 else numpy.ones(alignvals.shape[0], dtype=numpy.bool)
    view = healpix.ang2pix(view_resolution, numpy.deg2rad(alignvals[:, 1:3]))
    views = numpy.unique(view)
    _logger.error("# views: %d -- %s"%(len(views), str(view[:5])))
    maximum_views = 0
    if cull_overrep:
        vhist = numpy.histogram(view, len(views)+1)[0]
        _logger.error("hist: %s"%str(vhist))
        maximum_views = numpy.mean(vhist[vhist>0])
    for v in views:
        vsel = numpy.logical_and(sel, v==view)
        _logger.error("View: %d -- %d, %d, %d"%(v, numpy.sum(sel), numpy.sum(v==view), alignvals.shape[0]))
        vidx = numpy.argwhere(vsel).squeeze()
        cc = alignvals[vidx, 10]
        if cc.shape[0] == 0: continue
        if threshold_type == 1:
            cc_threshold = analysis.otsu(cc, threshold_bins)
        elif threshold_type == 3:
            cc_threshold = analysis.threshold_from_total(cc, cc_total, keep_low_cc)
        if maximum_views > 0:
            cc_threshold = analysis.threshold_max(cc, cc_threshold, maximum_views, keep_low_cc)
        csel = cmp(alignvals[:, 10], cc_threshold)
        vsel = numpy.logical_and(vsel, csel)
        sel[numpy.argwhere(numpy.logical_not(vsel)).squeeze()] = 0
        _logger.info("View: %d -> Kept %d of %d"%(v, numpy.sum(sel[vidx]), cc.shape[0]))
    return sel

def initialize(files, param):
    # Initialize global parameters for the script
    
    #param['spi'] = spider.open_session(files, **param)
    pass

def finalize(files, **extra):
    # Finalize global parameters for the script
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    
    if 1 == 1:
        from ..core.app.settings import setup_options_from_doc
        setup_options_from_doc(parser, classify_projections, group=pgroup)
    else:
        from ..core.app.settings import OptionGroup
        group = OptionGroup(parser, "Classification", "Options controlling classification", group_order=0,  id=__name__)
        group.add_option("",   threshold_type=('None', 'Auto', 'CC', 'Total'),       help="Type of thresholding to perform", default=0)
        group.add_option("",   cc_threshold=0.0,        help="Cross-correlation threshold value (used with `--threshold-type CC`")
        group.add_option("",   cc_total=0.9,            help="Total number of high cross-correlation projections to keep, if < 1.0, assumes a fraction, otherwise total number (used with `--threshold-type Total`)")
        group.add_option("",   threshold_by_view=False, help="Apply threshold within each view (determined with `view_resolution`)")
        group.add_option("",   view_resolution=1,       help="Resolution (or angular increment) to consider, e.g. 1 ~ 30 deg (theta-delta), 2 ~ 15 deg, 3 ~ 7 deg (used when `--threshold-by-view` is set True)")
        group.add_option("",   threshold_bins=0,        help="Number of bins to use in Otzus method: 0 = SQRT(total), -1 = total/16, otherwise use the number given (used with `--threshold-type Auto`)")
        group.add_option("",   keep_low_cc=False,       help="Set to True if you want to keep the low cross-correlation particles instead")
        group.add_option("",   cull_overrep=False,      help="Set to True if you want to ensure each view has roughly the same number of particles (used when `--threshold-by-view` is set True)")
        group.add_option("",   cc_nstd=3,               help="Remove low cross-correlation particles whose cross-correlation score is less than cc_nstd times the standard deviation outside the mean")
        pgroup.add_option_group(group)
        
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input alignment files", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for selection file", gui=dict(filetype="save"), required_file=True)
        parser.change_default(log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if options.threshold_type == 3:
        if options.cc_total <= 0: raise OptionValueError, "Invalid value for `--cc-total` %f; must be greater than 0"%options.cc_total
    if options.threshold_by_view:
        if options.view_resolution < 1: raise OptionValueError, "Invalid value for `--view-resolution` %d; must be greater than 0"%options.view_resolution
    if main_option:
        if options.threshold_type == 0: OptionValueError, "Invalid value for `--threshold-type` %d; cannot be 'None'"%options.threshold_type
        if not spider_utility.test_valid_spider_input(options.input_files):
            raise OptionValueError, "Multiple input files must have numeric suffix, e.g. vol0001.spi"

def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Classify projections based on an alignment
                        
                        http://
                        
                        $ %prog align1.spi align2.spi -o select01.spi --threshold-type Auto --threshold-by-view --view-resolution 2
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=False,
        use_version = False,
    )
def dependents(): return []
if __name__ == "__main__": main()

