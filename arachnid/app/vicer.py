''' Clean particle selection with the view classifier or ViCer

This script (`ara-vicer`) was designed to post clean an existing particle selection. It requires
that the projections be grouped by view as well as 2D alignment parameters. This can be obtained by
2D-reference free classification or 3D orientation determination.

Notes
=====

 #. Filenames: Must follow the SPIDER format with a number before the extension, e.g. mic_00001.spi. Output files just require the number of digits: `--output sndc_0000.spi`

 #. Parallel Processing - Several micrographs can be run in parallel (assuming you have the memory and cores available). `-p 8` will run 8 micrographs in parallel. 
 
 #. Supports both SPIDER and RELION alignment files - if the alignment comes from pySPIDER, you must specified --scale-spi or set True.

Examples
========

.. sourcecode :: sh
    
    # Run with a disk as a template on a raw film micrograph
    
    $ ara-vicer -i image_file_0001.spi -a align_file.spi -o output/clean_000.spi -p params.spi -w 10

Critical Options
================

.. program:: ara-vicer

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input stack filenames or single input stack filename template
    If you use the parameters `-i` or `--inputfiles` the filenames may be comma or 
    space separated on the command line; they must be comma seperated in a configuration 
    file. Note, these flags are optional for input files; the filenames must be separated 
    by spaces. For a very large number of files (>5000) use `-i "filename*"`

.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for the output embeddings and selection files (prefixed with sel_$output)

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Filename for SPIDER parameter file describing a Cryo-EM experiment

.. option:: -a <FILENAME>, --alignment <FILENAME> 
    
    Input file containing alignment parameters

Useful Options
===============

These options 

.. program:: ara-vicer

.. option:: --order <INT>
    
    Reorganize views based on their healpix order (overrides the resolution parameter)

.. option:: --prob-reject <FLOAT>
    
    Probablity that a rejected particle is bad

.. option:: --neig <INT>
    
    Number of eigen vectors to use

.. option:: --expected <FLOAT>
    
    Expected fraction of good data

.. option:: --resolution <FLOAT>
    
    Filter to given resolution - requires apix to be set

Customization Options
=====================

Generally, these options do not need to be changed, unless you are giving the program a non-standard input.

.. program:: ara-vicer
    
.. option:: --disable-rtsq <BOOL>

    Do not use alignment parameters to rotate projections in 2D
    
.. option:: --scale-spi <BOOL>

    Scale the SPIDER translation (if refinement was done by pySPIDER)
    
Testing Options
===============

Generally, these options do not need to be considered unless you are developing or testing the code.

.. program:: ara-vicer
    
.. option:: --single-view <INT>

    Test the algorithm on a specific view
    
.. option:: --random-view <INT>

    Set number of views to assign randomly, 0 means skip this
    
.. option:: --disable-bispec <BOOL>

    Disable bispectrum feature space
    
.. option:: --nsamples <INT>

    Number of rotational samples
    
.. option:: --angle-range <FLOAT>

    Angular search range

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by MPI-enabled scripts... <mpi-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`

.. Created on Sep 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.app import program
from ..core.image import ndimage_file
from ..core.image import ndimage_utility
from ..core.image import rotate
from ..core.image import ndimage_processor
from ..core.image import ndimage_interpolate
from ..core.image import preprocess_utility
from ..core.metadata import format
from ..core.metadata import spider_params
from ..core.metadata import format_alignment
from ..core.metadata import namedtuple_utility
from ..core.metadata import format_utility
from ..core.metadata import selection_utility
from ..core.parallel import mpi_utility, openmp
from ..core.orient import healpix
from ..core.orient import spider_transforms
from ..core.learn import unary_classification
from ..core.learn import dimensionality_reduction
import logging
import numpy
import scipy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(input_vals, output, **extra):#, neig=1, nstd=1.5
    '''Concatenate files and write to a single output file
        
    :Parameters:
        
        input_vals : list 
                     Tuple(view id, image labels and alignment parameters)
        output : str
                 Filename for output file
        extra : dict
                Unused key word arguments
                
    :Returns:
        
        filename : str
                   Current filename
    '''
    
    _logger.info("Processing view %d"%int(input_vals[0]))
    
    label, align = rotational_sample(*input_vals[1:], **extra)
    filename = label[0] if isinstance(label, tuple) else label[0][0]
    mask = create_mask(filename, **extra)
    
    openmp.set_thread_count(1) # todo: move to process queue
    data = ndimage_processor.create_matrix_from_file(label, image_transform, align=align, mask=mask, dtype=numpy.float32, **extra)
    openmp.set_thread_count(extra['thread_count'])
    assert(data.shape[0] == align.shape[0])
    tst = data-data.mean(0)

    feat = embed_sample(tst, **extra)
        
    rsel=None
    if feat is not None:
        sel, rsel, dist = one_class_classification(feat, **extra)
        if isinstance(label, tuple):
            filename, label = label
            format.write_dataset(output, numpy.hstack((sel[:, numpy.newaxis], dist[:, numpy.newaxis], align[:, 0][:, numpy.newaxis], label[:, 1][:, numpy.newaxis], align[:, (3,4,5,1)], feat)), input_vals[0], label, header='select,dist,rot,group,psi,tx,ty,mirror')
        else:
            format.write_dataset(output, numpy.hstack((sel[:, numpy.newaxis], dist[:, numpy.newaxis], align[:, 0][:, numpy.newaxis], align[:, (3,4,5,1)], feat)), input_vals[0], label, header='select,dist,rot,psi,tx,ty,mirror', default_format=format.csv)
        _logger.info("Finished embedding view: %d"%(int(input_vals[0])))
    else:
        _logger.info("Skipping view (too few projections): %d"%int(input_vals[0]))
    
    return input_vals, rsel

def embed_sample(samp, neig, expected, niter=5, **extra):
    ''' Embed the sample images into a lower dimensional factor space
    
    :Parameters:
        
        samp : array
               2D array where each row is an unraveled image and each column a pixel
        neig : int
               Number of Eigen vectors
        expected : float
                   Probability an image does not contain an outlier
        extra : dict
                Unused keyword arguments
    
    :Returns:
        
        feat : array
               2D array where each row is a compressed image and each column a factor
    '''
    
    eigv, feat=dimensionality_reduction.dhr_pca(samp, samp, neig, expected, True, niter)
    _logger.info("Eigen: %s"%(",".join([str(v) for v in eigv[:10]])))
    tc=eigv.cumsum()
    _logger.info("Eigen-cum: %s"%(",".join([str(v) for v in tc[:10]])))
    return feat

def one_class_classification(feat, neig, nsamples, prob_reject, **extra):
    '''Reject outliers using one-class classification based on the mahalanobis distance
    estimate from a robust covariance as calculated by minimum covariance determinant.
    
    :Parameters:
        
        feat : array
               2D array where each row is a compressed image and each column a factor
        neig : int
               Number of Eigen vectors
        nsamples : int
                   Number of images (before constrained rotation) 
        prob_reject : float
                      Probability threshold for rejecting outliers
        extra : dict
                Unused keyword arguments
    
    :Returns:
        
        sel : array
              Boolean array for each image
        rsel : array
               Boolean array for each group of rotated images
        dist : array
               Mahalanobis distance from the median for each image
    '''

    feat=feat[:, :neig]
    sel, dist = unary_classification.robust_mahalanobis_with_chi2(feat, prob_reject, True)
    #_logger.debug("Cutoff: %d -- for df: %d | sel: %d"%(cut, feat.shape[1], numpy.sum(sel)))
    if nsamples > 1:
        rsel = numpy.ones(int(feat.shape[0]/nsamples), dtype=numpy.bool)
        for i in xrange(rsel.shape[0]):
            rsel[i] = numpy.alltrue(sel[i*nsamples:(i+1)*nsamples])
    else: rsel = sel
    return sel, rsel, dist

def rotational_sample(label, align, nsamples, angle_range, **extra):
    ''' Generate alignment parameters and labels for each rotational sample
    
    :Parameters:
        
        label : array
                2D integer array: file ID, slice ID
        align : array
                2d float array for alignment parameters
        nsamples : int
                   Number of rotational samples
        angle_range : float
                      Allowed variance in in-plane rotation
        extra : dict
                Unused keyword arguments
    
    :Returns:
        
        label2 : array
                 2D integer array expanded for each rotation: file ID, slice ID
        align2 : array
                2d float array expanded for each rotation for alignment parameters
    '''
    
    if nsamples < 2:
        return label, align
    if isinstance(label, tuple):
        filename, label = label
        label2 = numpy.zeros((label.shape[0]*nsamples, label.shape[1]))
        align2 = numpy.zeros((align.shape[0]*nsamples, align.shape[1]))
        for i in xrange(len(label)):
            label2[i*nsamples:(i+1)*nsamples] = label[i]
            align2[i*nsamples:(i+1)*nsamples] = align[i]
            align2[i*nsamples:(i+1)*nsamples, 0]=scipy.linspace(-angle_range/2.0, angle_range/2.0, nsamples,True)
    else:
        label2 = []
        align2 = numpy.zeros((align.shape[0]*nsamples, align.shape[1]))
        for i in xrange(len(label)):
            label2.extend([label[i] for j in nsamples])
            align2[i*nsamples:(i+1)*nsamples] = align[i]
            align2[i*nsamples:(i+1)*nsamples, 0]=scipy.linspace(-angle_range/2.0, angle_range/2.0, nsamples,True)
    return label2, align2

def create_mask(filename, pixel_diameter, apix, **extra):
    ''' Create a disk mask from the input file size, diameter in pixels and target
    pixel size.
    
    :Parameters:
    
        filename : str
                   Input image file
        pixel_diameter : int
                         Diameter of mask in pixels
        apix : float
               Pixel spacing
        extra : dict
                Unused keyword arguments
    
    :Returns:
        
        mask : array
               2D array of disk mask
    '''
    
    img = ndimage_file.read_image(filename)
    bin_factor = decimation_level(apix, pixel_diameter=pixel_diameter, **extra)
    shape = numpy.asarray(img.shape, dtype=numpy.float)/bin_factor
    mask = ndimage_utility.model_disk(int(pixel_diameter/2.0/bin_factor), tuple(shape.astype(numpy.int)))
    return mask

def resolution_from_order(apix, pixel_diameter, order, resolution, **extra):
    ''' Estimate a target resolution based on the angular increment. Returns
    new estimate only if less-resolved than the current.
    
    :Parameters:
        
        apix : float
               Pixel spacing
        pixel_diameter : int
                         Diameter of mask in pixels
        order : int
                Healpix order
        resolution : float
                     Current resolution
        extra : dict
                Unused keyword arguments
    
    :Returns:
        
        resolution : float
                     Estimated resolution
    '''
    
    if order == 0 or 1 == 1: return resolution
    res = numpy.tan(healpix.nside2pixarea(order))*pixel_diameter*apix
    if res > resolution: resolution=res
    return resolution

def order_from_resolution(apix, pixel_diameter, resolution, **extra):
    ''' Estimate healpix order from resolution
    
    :Parameters:
        
        apix : float
               Pixel spacing
        pixel_diameter : int
                         Diameter of mask in pixels
        resolution : float
                     Current resolution
        extra : dict
                Unused keyword arguments
    
    :Returns:
        
        order : int
                Healpix order
    '''
    
    theta_delta = numpy.rad2deg( numpy.arctan( resolution / (pixel_diameter*apix) ) )
    _logger.info("Target sampling %f for resolution %f -> %d"%(theta_delta, resolution, healpix.theta2nside(numpy.deg2rad(theta_delta))))
    return healpix.theta2nside(numpy.deg2rad(theta_delta))

def decimation_level(apix, window, **extra):
    '''
    :Parameters:
        
        apix : float
               Pixel spacing
        window : int
                 Size of the window in pixels
        extra : dict
                Unused keyword arguments
    
    :Returns:
        
        out : float
              Target downsampling factor
    '''
    
    resolution = resolution_from_order(apix, **extra)
    dec = resolution / (apix*3)
    d = float(window)/dec + 10
    d = window/float(d)
    return min(max(d, 1), 8)

def image_transform(img, i, mask, resolution, apix, var_one=True, align=None, disable_bispec=False, disable_rtsq=False, scale_spi=False, **extra):
    ''' Transform the image
    
    :Parameters:
        
        img : array
              2D image matrix
        i : int
            Offset into alignment parameters 
        mask : array
               2D array of disk mask
        resolution : float
                     Target resolution of image 
        apix : float
               Pixel spacing 
        var_one : bool
                  Normalize image to variance 1 
        align : array
                2D array of alignment parameters
        disable_bispec : bool
                         Do not estimate bispectra of image
        disable_rtsq : bool
                       Disable rotate/translate image
        scale_spi : bool
                    Scale translations before rotate/translate
        extra : dict
                Unused keyword arguments
    
    :Returns:
        
        out : array
              1D representation of the image
    '''
    
    if not disable_rtsq: 
        if scale_spi:
            img = rotate.rotate_image(img, align[i, 3], align[i, 4]/apix, align[i, 5]/apix)
        else:
            img = rotate.rotate_image(img, align[i, 3], align[i, 4], align[i, 5])
    elif align[i, 0] != 0: img = rotate.rotate_image(img, -align[i, 0])
    if align[i, 1] > 179.999: img = ndimage_utility.mirror(img)
    ndimage_utility.vst(img, img)
    bin_factor = decimation_level(apix, resolution=resolution, **extra)
    if bin_factor > 1: img = ndimage_interpolate.downsample(img, bin_factor)
    if mask is not None:
        ndimage_utility.normalize_standard_norm(img, mask, var_one, out=img)
    
    if not disable_bispec:
        img, freq = ndimage_utility.bispectrum(img, int(img.shape[0]-1), 'uniform')#, scale='unbiased'
        '''img /= numpy.pow(img.real, 2/3)
        set freq0 = 0
        '''
        freq;
        img = numpy.log10(numpy.abs(img.real)+1)
    
    return img

def group_by_reference(label, align, ref):
    ''' Group alignment entries by view number
    
    :Parameters:
        
        label : array
                2D integer array where rows are images and columns are file ID, slice ID
        align : array
                2D float array for alignment parameters where rows are images and columns alignment parameters
        label : array
                1D integer array containing the group number for each image
    
    :Returns:
        
        group : list
                List of tuples (view, selected label, selected align)
    '''
    
    group=[]
    refs = numpy.unique(ref)
    if isinstance(label, tuple):
        filename, label = label
        _logger.info("Processing %d projections from %d stacks grouped into %d views"%(len(label), len(numpy.unique(label[:, 0])), len(refs)))
        for r in refs:
            sel = r == ref
            group.append((r, (filename, label[sel]), align[sel]))
    else:
        stack_count = {}
        for i in xrange(len(label)):
            if label[i][0] not in stack_count: 
                stack_count[label[i][0]]=1 
        stack_count = len(stack_count)
        _logger.info("Processing %d projections from %d stacks grouped into %d views - no spi"%(len(label), stack_count, len(refs)))
        for r in refs:
            sel = r == ref
            group.append((r, [label[i] for i in numpy.argwhere(sel).squeeze()], align[sel]))
    return group

def read_alignment(files, alignment="", disable_mirror=False, order=0, random_view=0, diagnostic="", class_index=0, **extra):
    ''' Read alignment parameters
    
    :Parameters:
        
        files : list
                List of input files containing particle stacks
        alignment : str
                    Input filename containing alignment parameters
        disable_mirror : bool
                         Flag to disable mirroring
        order : int
                Healpix resolution
        random_view : int
                      Assign projections to given number of random views (0 disables)
        diagnostic : str
                     Output file for diagnostic view average stack
        class_index : int
                      Class to select
        extra : dict
                Unused keyword arguments
    
    :Returns:
    
        files : list
                List of filename, index tuples
        align : array
                2D array of alignment parameters
        ref : array
              1D array of view groups
    '''

    numpy.seterr(all='raise')
    files, align = format_alignment.read_alignment(alignment, files[0], use_3d=False, align_cols=8, class_index=class_index)
    align[:, 7]=align[:, 6]
    
    if order > 0: spider_transforms.coarse_angles(order, align, half=not disable_mirror, out=align)
    print 'after', len(numpy.unique(align[:, 6]))
        
    if diagnostic != "":
        _logger.info("Preprocessing diagnositic images")
        thread_count = extra['thread_count']
        extra['thread_count']=extra['worker_count']
        extra['disable_bispec']=True
        extra['resolution']=1.0
        #avg = ndimage_processor.image_array_from_file(files, preprocess_utility.phaseflip_align2d_i, param=align, **extra)
        avg = ndimage_processor.image_array_from_file(files, image_transform, mask=None, align=align, order=order, thread_num=thread_count, **extra)
        _logger.info("Preprocessing diagnositic images - finished")
        ref = align[:, 6].astype(numpy.int)
        view = numpy.unique(ref)
        _logger.info("Averaging %d views from %d - %d"%(len(view), view.min(), view.max()))
        avgs = []
        _logger.info("Averaging diagnositic images")
        for i, v in enumerate(view):
            if numpy.sum(v==ref)==0: continue
            avgs.append(avg[v==ref].mean(axis=0))
        _logger.info("Averaging diagnositic images - finished")
        ndimage_file.write_stack(diagnostic, avgs)
        
        if 1 == 0:
            _logger.info("Averaging diagnositic images - unmirrored")
            for i, v in enumerate(view):
                sel = numpy.logical_and(v==ref, align[:, 1] <= 90.0)
                if numpy.sum(sel)==0: continue
                avgs.append(avg[sel].mean(axis=0))
            _logger.info("Averaging diagnositic images - finished")
            ndimage_file.write_stack(format_utility.add_prefix(diagnostic, 'unmir_'), avgs)
            
            
            _logger.info("Averaging diagnositic images - mirrored")
            for i, v in enumerate(view):
                sel = numpy.logical_and(v==ref, align[:, 1] > 90.0)
                if numpy.sum(sel)==0: continue
                avgs.append(avg[sel].mean(axis=0))
            _logger.info("Averaging diagnositic images - finished")
            ndimage_file.write_stack(format_utility.add_prefix(diagnostic, 'mir_'), avgs)
    
    
    if random_view>0:
        ref = numpy.random.randint(0, random_view, len(align))
    else:
        ref = align[:, 6].astype(numpy.int)
    return files, align, ref

def init_root(files, param):
    # Initialize global parameters for the script
    
    spider_params.read(param['param_file'], param)
    if param['order'] < 0:
        param['order'] = order_from_resolution(**param)
    assert('apix' in param)
    assert('window' in param)
    if param['scale_spi']: _logger.info("Scaling translations by pixel size (pySPIDER input)")
    if param['resolution'] > 0.0: _logger.info("Filter and decimate to resolution %f by a factor of %f"%(resolution_from_order(**param), decimation_level(**param)))
    if not param['disable_rtsq']: _logger.info("Rotate and translate data stack")
    _logger.info("Rejection precision: %f"%param['prob_reject'])
    _logger.info("Number of Eigenvalues: %f"%param['neig'])
    if param['order'] > 0: _logger.info("Angular order %f sampling %f degrees "%(param['order'], healpix.nside2pixarea(param['order'], True)))
    #_logger.info("nsamples: %f"%param['nsamples'])
        
    param['sel_by_mic']={}
    _logger.info("Reading alignment file and grouping projections")
    group = group_by_reference(*read_alignment(files, **param))
    _logger.info("Created %d groups"%len(group))
    if param['single_view'] > 0:
        _logger.info("Using single view: %d"%param['single_view'])
        tmp=group
        group = [tmp[param['single_view']-1]]
    else:
        count = numpy.zeros((len(group)))
        for i in xrange(count.shape[0]):
            if isinstance(group[i][1], tuple):
                count[i] = len(group[i][1][1])
            else:
                count[i] = len(group[i][1])
        index = numpy.argsort(count)
        newgroup=[]
        for i in index[::-1]:
            if count[i] > 20:
                _logger.info("Group: %d = %d"%(i, count[i]))
                newgroup.append(group[i])
        group=newgroup
    _logger.info("Processing %d groups - after removing views with less than 20 particles"%len(group))
    return group

def update_selection_dict(sel_by_mic, label, sel):
    ''' Maps selections from view to stack in a dictionary
    
    :Parameters:
        
        sel_by_mic : dict
                     Dictionary to update
        label : tuple or list 
                If tuple (filename, label), otherwise list of tuples [(filename, index)]
        sel : array
              Boolean array defining selections
    '''
    
    if isinstance(label, tuple):
        filename, label = label
        for i in numpy.argwhere(sel):
            sel_by_mic.setdefault(int(label[i, 0]), []).append(int(label[i, 1]+1))
    else:
        for i in numpy.argwhere(sel):
            sel_by_mic.setdefault(label[i][0], []).append(int(label[i][1])+1)

def initialize(files, param):
    # Initialize global parameters for the script
    
    if not mpi_utility.is_root(**param): 
        spider_params.read(param['param_file'], param)

def reduce_all(val, sel_by_mic, id_len=0, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    _logger.info("Reducing to root selections")
    input, sel = val
    label = input[1]
    update_selection_dict(sel_by_mic, label, sel) 
    tot=numpy.sum(sel)
    total = len(label[1]) if isinstance(label, tuple) else len(label)
    return "%d - Selected: %d -- Removed %d"%(input[0], tot, total-tot)

def finalize(files, output, sel_by_mic, finished, nsamples, thread_count, neig, input_files, alignment, diagnostic, **extra):
    # Finalize global parameters for the script
    
    nsamples = None
    
    for filename in finished:
        label = filename[1]
        data = format.read(output, numeric=True, spiderid=int(filename[0]))
        feat, header = namedtuple_utility.tuple2numpy(data)
        off = header.index('mirror')+1
        if nsamples is None:
            nsamples = len(numpy.unique(feat[:, 4]))
            _logger.info("Number of samples per view: %d"%nsamples)
        feat = feat[:, off:]
        #sel, rsel, dist = one_class_classification(feat, **extra)
        sel, rsel = one_class_classification(feat, neig=neig, nsamples=nsamples, **extra)[:2]
        _logger.debug("Read %d samples and selected %d from finished view: %d"%(feat.shape[0], numpy.sum(rsel), int(filename[0])))
        for j in xrange(len(feat)):
            data[j] = data[j]._replace(select=sel[j])
        format.write(output, data, spiderid=int(filename[0]))
        update_selection_dict(sel_by_mic, label, rsel)
    tot=0
    for id, sel in sel_by_mic.iteritems():
        n=len(sel)
        tot+=n
        _logger.debug("Writing %d to selection file %d"%(n, id))
        sel = numpy.asarray(sel)
        format.write(output, numpy.vstack((sel, numpy.ones(sel.shape[0]))).T, prefix="sel_", spiderid=id, header=['id', 'select'], default_format=format.spidersel)
    
    data = format.read(alignment, numeric=True)
    data = selection_utility.select_stack_subset(data, sel_by_mic)
    if len(data) > 0 and hasattr(data[0], 'rlnImageName'):
        format.write(output, data, nospiderid=True, format=format.star)
    else:
        format.write(output, data, nospiderid=True, format=format.spiderdoc)
    if diagnostic != "":
        extra['thread_count']=extra['worker_count']
        filename = finished[0][1] if isinstance(finished[0], tuple) else finished[0]
        filename = filename[0] if isinstance(filename, tuple) else filename
        files, align = format_alignment.read_alignment(data, image_file=filename, use_3d=False, align_cols=8)
        align[:, 7]=align[:, 6]
        order=extra['order']
        if order > 0: spider_transforms.coarse_angles(order, align, half=not extra['disable_mirror'], out=align)
        avg = ndimage_processor.image_array_from_file(files, preprocess_utility.align2d_i, param=align, **extra)
        ref = align[:, 6].astype(numpy.int)
        view = numpy.unique(ref)
        avgs = []
        for i, v in enumerate(view):
            if numpy.sum(v==ref)==0: continue
            avgs.append(avg[v==ref].mean(axis=0))
        ndimage_file.write_stack(format_utility.add_prefix(diagnostic, 'sel_'), avgs)
    
    _logger.info("Selected %d projections"%(tot))
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "ViCer", "Particle verification with View Classifier or ViCer",  id=__name__)
    group.add_option("", nsamples=1,                help="Number of rotational samples")
    group.add_option("", angle_range=3.0,           help="Angular search range")
    group.add_option("", resolution=40.0,           help="Filter to given resolution - requires apix to be set")
    group.add_option("", disable_rtsq=False,        help="Do not use alignment parameters to rotate projections in 2D")
    group.add_option("", neig=2,                    help="Number of eigen vectors to use", dependent=False)
    group.add_option("", expected=0.8,              help="Expected fraction of good data", dependent=False)
    group.add_option("", scale_spi=False,           help="Scale the SPIDER translation (if refinement was done by pySPIDER")
    group.add_option("", single_view=0,             help="Test the algorithm on a specific view")
    group.add_option("", disable_bispec=False,      help="Disable bispectrum feature space")
    group.add_option("", order=0,                   help="Reorganize views based on their healpix order (overrides the resolution parameter)")
    group.add_option("", prob_reject=0.97,          help="Probablity that a rejected particle is bad", dependent=False)
    group.add_option("", random_view=0,             help="Set number of views to assign randomly, 0 means skip this")
    group.add_option("", disable_mirror=False,      help="Disable mirroring and consider the full sphere in SO2")
    group.add_option("", niter=5,                   help="Number of iterations for cleaning")
    group.add_option("", diagnostic="",             help="Diagnosic view averages", gui=dict(filetype="save"), dependent=False)
    group.add_option("", class_index=0,             help="Select a specifc class within the alignment file")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input particle stacks, e.g. cluster/win/win_*.dat ", required_file=True, gui=dict(filetype="open"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with with no digits at the end (e.g. this is bad -> sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-a", alignment="",   help="Input file containing alignment parameters", required_file=True, gui=dict(filetype="open"))
        spider_params.setup_options(parser, pgroup, True)

def check_options(options, main_option=False):
    #Check if the option values are valid
#    from ..core.app.settings import OptionValueError

    if options.nsamples < 1: options.nsamples=1

def main():
    #Main entry point for this script
    program.run_hybrid_program(__name__,
        description = '''Clean a particle selection of any remaning bad windows
                        
                        Example:
                         
                        $ %prog input-stack.spi -o view_001.dat -p params.spi -a alignment.spi -w 4
                        
                      ''',
        use_version = True,
        supports_OMP=True,
    )

def dependents(): return []
if __name__ == "__main__": main()

