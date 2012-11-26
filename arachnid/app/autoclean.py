'''
.. Created on Sep 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import matplotlib
matplotlib.use("Agg")

from ..core.app.program import run_hybrid_program
#from ..core.image.ndplot import pylab
from ..core.image import ndimage_file, eman2_utility, analysis, ndimage_utility, reconstruct
from ..core.metadata import spider_utility, format, format_utility, spider_params
from ..core.parallel import mpi_utility, process_queue
from ..core.image import manifold
from arachnid.core.util import plotting #, fitting
import logging, numpy, os, scipy,itertools

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(input_vals, input_files, output, write_view_stack=0, sort_view_stack=False, id_len=0, **extra):
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
    id_len : int
             Max length of SPIDER ID
    extra : dict
            Unused key word arguments
                
    :Returns:
    
    filename : str
               Current filename
    peaks : str
            Coordinates found
    '''
    
    output = spider_utility.spider_filename(output, input_vals[0], id_len)
    data, mask, avg, template = read_data(input_files, *input_vals[1:3], **extra)
    eigs, sel, energy = classify_data(data, None, output=output, view=input_vals[0], **extra)
    
    write_dataset(output, eigs, sel, input_vals[0])
    
    plot_examples(input_files, input_vals[1], output, eigs, sel, **extra)
    
    avg3 = []
    avg3.append( comput_average(input_files, input_vals[1], input_vals[2], subset=sel, **extra) )
    avg3.append( comput_average(input_files, input_vals[1], input_vals[2], subset=numpy.logical_not(sel), **extra) )
    
    return input_vals, eigs, sel, avg3[:3], "Energy: %f - Num Eigen: %d"%energy

def classify_data(data, test=None, neig=1, thread_count=1, resample=0, sample_size=0, local_neighbors=0, min_group=None, view=0, **extra):
    ''' Classify the aligned projection data grouped by view
    
    :Parameters:
    
    data : array
           2D array where each row is an aligned and transformed image
    output : str
             Output filename for intermediary data
    neig : float
           Number of eigen vectors to use (or mode)
    thread_count : int
                    Number of threads
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    eigs : array
           2D array of embedded images
    sel : array
          1D array of selected images
    '''
    
    #from sklearn import mixture
    
    train = process_queue.recreate_global_dense_matrix(data)
    test = train
    eigs, idx, vec, energy = analysis.pca(train, test, neig, test.mean(axis=0))
    sel = one_class_classification(eigs)
    return eigs, sel, (energy, idx)

def one_class_classification(feat):
    ''' Perform one-class classification using Otsu's algorithm
    
    :Parameters:
    
    feat : array
           Feature space
    
    :Returns:
    
    sel : array
          Selected samples
    '''
    
    feat = feat.copy()
    cent = numpy.median(feat, axis=0)
    dist_cent = scipy.spatial.distance.cdist(feat, cent.reshape((1, len(cent))), metric='euclidean').ravel()
    sel = analysis.robust_rejection(dist_cent)
    return sel

def comput_average(input_files, label, align, subset=None, use_rtsq=False, **extra):
    '''
    '''
    
    if subset is not None:
        label = label[subset]
        align = align[subset]
    
    avg = None
    for i, img in enumerate(ndimage_file.iter_images(input_files, label)):
        m = align[i, 1] > 179.999
        if use_rtsq: img = eman2_utility.rot_shift2D(img, align[i, 5], align[i, 6], align[i, 7], m)
        elif m:      img[:,:] = eman2_utility.mirror(img)
        if avg is None: avg = numpy.zeros(img.shape)
        avg += img
    avg /= (i+1)
    return avg

def classify_data2(data, ref, test=None, neig=1, thread_count=1, resample=0, sample_size=0, local_neighbors=0, min_group=None, view=0, **extra):
    ''' Classify the aligned projection data grouped by view
    
    :Parameters:
    
    data : array
           2D array where each row is an aligned and transformed image
    output : str
             Output filename for intermediary data
    neig : float
           Number of eigen vectors to use (or mode)
    thread_count : int
                    Number of threads
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    eigs : array
           2D array of embedded images
    sel : array
          1D array of selected images
    '''
    
    sel = None
    iter = 1
    for i in xrange(iter):
        if resample > 0:
            test = process_queue.recreate_global_dense_matrix(data)
            train = analysis.resample(data, resample, sample_size, thread_count)
        elif local_neighbors > 0:
            test = process_queue.recreate_global_dense_matrix(data)
            if ref is not None:
                train = numpy.empty_like(test)
                refs = numpy.unique(ref)
                offset = 0
                for r in refs:
                    train1 = manifold.local_neighbor_average(test[r==ref], manifold.knn(test[r==ref], local_neighbors))
                    end = offset+train1.shape[0]
                    train[offset:end] = train1
                    offset = end
            else:
                train = manifold.local_neighbor_average(test, manifold.knn(test, local_neighbors))
        else: 
            train = process_queue.recreate_global_dense_matrix(data)
            test = train
            train = train[:min_group]
        if sel is not None: train = train[sel]
        
        if 1 == 0:
            eigs, evals, index = manifold.diffusion_maps(test, 2, k=15, mutual=False, batch=10000)
            if index is not None:
                eigs2 = eigs
                eigs = numpy.zeros((test.shape[0], eigs2.shape[1]))
                eigs[index.squeeze()] = eigs2
            _logger.info("Eigen values: %s"%",".join([str(v) for v in evals]))
            if index is not None: _logger.info("Subset: %d of %d"%(index.shape[0], test.shape[0]))
            idx = 0
            energy=0
        elif 1 == 0:
            from sklearn import decomposition
            ica = decomposition.FastICA()
            eigs = ica.fit(train).transform(test)[:, :2]
            idx = 0
            energy=0
        else:
            eigs, idx, vec,energy = analysis.pca(train, test, neig, test.mean(axis=0))
        eigs2 = eigs.copy()
        #eigs2 -= eigs2.min(axis=0)
        #eigs2 /= eigs2.max(axis=0)
        cent = numpy.median(eigs2, axis=0)
        eig_dist_cent = scipy.spatial.distance.cdist(eigs2, cent.reshape((1, len(cent))), metric='euclidean').ravel()
        th = analysis.otsu(eig_dist_cent)
        _logger.info("Threshold(%d)=%f"%(view, th))
        sel = eig_dist_cent < th
        
        if 1 == 1:
            nsel = numpy.logical_not(sel)
            n = min(numpy.sum(nsel), numpy.sum(sel))
            if train.shape[0] != test.shape[0]: train = test
            penergy = analysis.pca_train(train[numpy.argwhere(sel).squeeze()[:n]], neig)[2]
            nenergy = analysis.pca_train(train[numpy.argwhere(nsel).squeeze()[:n]], neig)[2]
            _logger.info("Energy: %f -> p: %f | n: %f"%(energy, penergy, nenergy))
        
        if 1 == 0:
            feat, evals, index = manifold.diffusion_maps(test, 5, k=10, mutual=True, batch=10000)
            '''
            if index is not None:
                index = numpy.argwhere(sel).squeeze()[index]
            else: index = numpy.argwhere(sel).squeeze()
            '''
            if index is not None:
                feat_old = feat
                feat = numpy.zeros((eigs2.shape[0], feat.shape[1]))
                try:
                    feat[index.squeeze()] = feat_old
                except:
                    _logger.error("%s != %s - %s"%(str(feat.shape), str(feat_old.shape), str(index.shape)))
                    raise
            _logger.info("Eigen values: %s"%",".join([str(v) for v in evals]))
            eigs = numpy.hstack((eigs, feat))
        if iter > 1: _logger.info("Selected: %d - %f"%(numpy.sum(sel), energy))
        assert(eigs.shape[0]==test.shape[0])
    return eigs, sel, (energy, idx)



def image_transform(img, idx, align, mask, hp_cutoff, use_rtsq=False, template=None, resolution=0.0, apix=None, disable_bispec=False, use_radon=False, bispec_window='gaussian', bispec_biased=False, bispec_lag=0.0081, bispec_mode=0, flip_mirror=False, pixel_diameter=None, **extra):
    ''' Transform an image
    
    .. todo:: add ctf correction
    
    :Parameters:
    
    img : array
          Image data
    idx : int
          Offset in the alignment array
    align : array
            Alignment parameters
    mask : array
           Mask
    hp_cutoff : float
                Highpass cutoff
    use_rtsq : bool
               Set true to rotate and translate
    template : array
               Template for translational alignment
    resolution : float
                 Resolution to filter data
    apix : float
           Pixel size
    disable_bispec : bool
                     Disable the bispectrum representation
    use_radon : bool
                Use radon transform
    bispec_window : str
                    Type of window to use for bispectrum
    bispec_biased : bool
                    Set true to estimate biased bispectrum
    bispec_lag : float
                 Percentage of the maximum lag
    pixel_diameter : int
                     Diameter of particle in pixels
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    out : array
          1D transformed image
    '''
    
    var_one=True
    freq=None
    align = align[idx]
    m = align[1] > 179.9999 if not flip_mirror else align[1] < 179.9999
    if use_rtsq: img = eman2_utility.rot_shift2D(img, align[5], align[6], align[7], m)
    elif m:      img = eman2_utility.mirror(img)
    if disable_bispec:
        img = eman2_utility.gaussian_high_pass(img, hp_cutoff, True)
    
    bin_factor = max(1, min(8, resolution / (apix*2))) if resolution > (2*apix) else 1
    if bin_factor > 1: img = eman2_utility.decimate(img, bin_factor)
    ndimage_utility.normalize_standard(img, mask, var_one, img)
    if not disable_bispec:
        ndimage_utility.normalize_standard(img, mask, var_one, img)
        scale = 'biased' if bispec_biased else 'unbiased'
        bispec_lag = int(bispec_lag*img.shape[0])
        bispec_lag = min(max(1, bispec_lag), img.shape[0]-1)
        #img *= mask
        try:
            img, freq = ndimage_utility.bispectrum(img, bispec_lag, bispec_window, scale) # 1.0 = apix
        except:
            _logger.error("%d, %d"%(img.shape[0], bispec_lag))
            raise
        
        idx = numpy.argwhere(numpy.logical_and(freq >= hp_cutoff, freq <= apix/resolution))
        if bispec_mode == 1: 
            img = img.real
            img = img[idx[:, numpy.newaxis], idx]
        elif bispec_mode == 2: 
            img = numpy.mod(numpy.angle(img), 2*numpy.pi) #img.imag
            img = img[idx[:, numpy.newaxis], idx]
        else:
            img = img[idx[:, numpy.newaxis], idx]
            img = numpy.hstack((img.real.ravel()[:, numpy.newaxis], img.imag.ravel()[:, numpy.newaxis]))
        #numpy.log10(img, img)
        ndimage_utility.normalize_standard(img, None, var_one, img)
    elif use_radon:
        img = ndimage_utility.frt2(img*mask)
    else:
        img = ndimage_utility.compress_image(img, mask)
    return img

def read_data(input_files, label, align, shift_data=0, **extra):
    ''' Read images from a file and transform into a matrix
    
    :Parameters:
    
    input_files : list
                  List of input file stacks
    label : array
            2D array of particle labels (stack_id, particle_id)
    align : array
            2D array of alignment parameters
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    data : array
           2D array of transformed image data, each row is a transformed image
    '''
    
    extra['hp_cutoff'] = extra['apix']/120 if extra['resolution_hi'] > 0.0 else 2.0/extra['pixel_diameter']
    mask, template = None, None
    if 'mask' not in extra: 
        mask, average = create_tightmask(input_files, label, **extra)
        if shift_data > 0:
            template = average
            for i in xrange(1, shift_data):
                template = create_template(input_files, label, template, **extra)
        resolution = extra['resolution']
        apix = extra['apix']
        bin_factor = min(8, resolution / (apix*2) ) if resolution > (apix*2) else 1.0
        if bin_factor > 1: extra['mask'] = eman2_utility.decimate(mask, bin_factor)
        if shift_data > 0:
            if bin_factor > 1: extra['template'] = eman2_utility.decimate(template, bin_factor)
        else: extra['template']=None
    data = ndimage_file.read_image_mat(input_files, label, image_transform, shared=True, align=align, **extra)
    mat = process_queue.recreate_global_dense_matrix(data)
    mat -= mat.min(axis=0)
    mat /= mat.max(axis=0)
    assert(numpy.alltrue(numpy.isfinite(mat)))
    
    return data, mask, average, template

def create_template(input_files, label, template, resolution, apix, hp_cutoff, pixel_diameter, **extra): #extra['pixel_diameter']
    ''' Create a tight mask from a view average
    
    :Parameters:
    
    input_files : list
                  List of input file stacks
    label : array
            2D array of particle labels (stack_id, particle_id)
    template : array
               Averaged view
    resolution : float
                 Resolution to filter data
    apix : float
           Pixel size
    pixel_diameter : int
                     Diameter of particle in pixels
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    mask : array
           2D array image mask
    '''
    
    avg2 = None
    for img2 in ndimage_file.iter_images(input_files, label):
        if avg2 is None: avg2 = numpy.zeros(img2.shape)
        avg2 += img2
    avg2 /= len(label)
    avg2 = eman2_utility.gaussian_low_pass(avg2, apix/resolution, True)
    avg2 = eman2_utility.gaussian_high_pass(avg2, hp_cutoff, True)
    return avg2

def create_tightmask(input_files, label, resolution, apix, hp_cutoff, **extra):
    ''' Create a tight mask from a view average
    
    :Parameters:
    
    input_files : list
                  List of input file stacks
    label : array
            2D array of particle labels (stack_id, particle_id)
    resolution : float
                 Resolution to filter data
    apix : float
           Pixel size
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    mask : array
           2D array image mask
    '''
    
    avg2 = None
    for img2 in ndimage_file.iter_images(input_files, label):
        if avg2 is None: avg2 = numpy.zeros(img2.shape)
        avg2 += img2
    avg2 /= len(label)
    temp = eman2_utility.gaussian_low_pass(avg2, apix/80.0, True)
    mask = ndimage_utility.segment(temp)
    avg2 = eman2_utility.gaussian_low_pass(avg2, apix/resolution, True)
    avg2 = eman2_utility.gaussian_high_pass(avg2, hp_cutoff, True)
    return scipy.ndimage.morphology.binary_fill_holes(mask), avg2

def plot_examples(filename, label, output, eigs, sel, ref=None, dpi=200, **extra):
    ''' Plot the manifold with example images
    
    :Parameters:
    
    filename : str
               Stack filename
    label : array
            Stack label ids
    output : str
             Output filename
    eigs : array
           Eigen embedding
    sel : array
          Boolean selection
    dpi : int
          Figure resolution
    '''
    
    if eigs.ndim == 1: return
    if eigs.shape[1] == 1: return
    filename = filename[0]
    image_size=0.4
    radius=20
    
    
    select = numpy.argwhere(sel < 0.5)
    cent = numpy.median(eigs, axis=0)
    rad = scipy.spatial.distance.cdist(eigs, cent.reshape((1, len(cent))), metric='euclidean').ravel()
    diff = eigs-cent
    theta = numpy.arctan2(diff[:,1], diff[:, 0])
    fig, ax = plotting.plot_embedding(rad, theta, select, ref, dpi=dpi)
    fig.savefig(format_utility.new_filename(output, "polar_", ext="png"), dpi=dpi)
    
    x=numpy.arange(1, len(eigs)+1)
    try:
        fig, ax = plotting.plot_embedding(x, eigs[:, 0], select, ref, dpi=dpi)
    except:
        _logger.error("%s -- %s"%(str(x.shape), str(eigs[:, 0].shape)))
        raise
    vals = plotting.nonoverlapping_subset(ax, x[select], eigs[select, 0], radius, 100)
    if len(vals) > 0:
        try:
            index = select[vals].ravel()
        except:
            _logger.error("%d-%d | %d"%(numpy.min(vals), numpy.max(vals), len(select)))
        else:
            #_logger.error("selected: %s"%str(index.shape))
            iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, label[index].squeeze()))
            plotting.plot_images(fig, iter_single_images, x[index], eigs[index, 0], image_size, radius)
    fig.savefig(format_utility.new_filename(output, "one_", ext="png"), dpi=dpi)
    
    fig, ax = plotting.plot_embedding(eigs[:, 0], eigs[:, 1], select, ref, dpi=dpi)
    vals = plotting.nonoverlapping_subset(ax, eigs[select, 0], eigs[select, 1], radius, 100)
    if len(vals) > 0:
        try:
            index = select[vals].ravel()
        except:
            _logger.error("%d-%d | %d"%(numpy.min(vals), numpy.max(vals), len(select)))
        else:
            #_logger.error("selected: %s"%str(index.shape))
            iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, label[index].squeeze()))
            plotting.plot_images(fig, iter_single_images, eigs[index, 0], eigs[index, 1], image_size, radius)
    fig.savefig(format_utility.new_filename(output, "neg_", ext="png"), dpi=dpi)
    
    select = numpy.argwhere(sel > 0.5)
    fig, ax = plotting.plot_embedding(eigs[:, 0], eigs[:, 1], select, ref, dpi=dpi)
    vals = plotting.nonoverlapping_subset(ax, eigs[select, 0], eigs[select, 1], radius, 100)
    if len(vals) > 0:
        try:
            index = select[vals].ravel()
        except:
            _logger.error("%d-%d | %d"%(numpy.min(vals), numpy.max(vals), len(select)))
        else:
            if index.shape[0] > 1:
                iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, label[index].squeeze()))
                plotting.plot_images(fig, iter_single_images, eigs[index, 0], eigs[index, 1], image_size, radius)
    fig.savefig(format_utility.new_filename(output, "pos_", ext="png"), dpi=dpi)

def write_dataset(output, eigs, sel, id):
    '''
    '''
    
    if eigs.ndim == 1: eigs = eigs.reshape((eigs.shape[0], 1))
    feat = numpy.hstack((sel[:, numpy.newaxis], eigs))
    label = numpy.zeros((len(eigs), 2))
    label[:, 0] = id
    label[:, 1] = numpy.arange(1, len(eigs)+1)
    format.write_dataset(os.path.splitext(output)[0]+".csv", feat, id, label, prefix="embed_", header="best")
    
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
        label[:, 1] = align[:, 4].astype(numpy.int)-1
        if numpy.max(label[:, 1]) >= ndimage_file.count_images(files[0]):
            label[:, 1] = numpy.arange(0, len(align))
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
                label[:, 1]-=1
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
                    label[:, 1] = numpy.arange(0, len(align[total:end]))
                total = end
            align = numpy.asarray(alignvals)
    ref = align[:, refidx].astype(numpy.int)
    refs = numpy.unique(ref)
    
    group = []
    for r in refs:
        sel = r == ref
        group.append((r, label[sel], align[sel], numpy.argwhere(sel)))
    return group, align, header

def initialize(files, param):
    # Initialize global parameters for the script
    
    if mpi_utility.is_root(**param):
        if param['resolution'] > 0.0: _logger.info("Filter and decimate to resolution: %f"%param['resolution'])
        if param['use_rtsq']: _logger.info("Rotate and translate data stack")
    
    group = None
    if mpi_utility.is_root(**param): 
        group, align, header = read_alignment(files, **param)
        #total = len(align)
        #param['min_group'] = numpy.min([len(g[1]) for g in group])
        param['alignheader'] = header
        param['alignment'] = align
        param['selected'] = numpy.zeros(len(align), dtype=numpy.bool)
        _logger.info("Cleaning bad particles from %d views"%len(group))
    group = mpi_utility.broadcast(group, **param)
    if param['single_view'] > 0:
        tmp=group
        group = [tmp[param['single_view']-1]]
    param['total'] = numpy.zeros((len(group), 3))
    param['sel_by_mic'] = {}
    
    return group

def reduce_all(val, input_files, total, sel_by_mic, output, file_completed, selected, id_len=0, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    input, eigs, sel, avg3, msg = val
    label = input[1]
    index = input[2]
    selected[index] = sel
    file_completed -= 1
    total[file_completed, 0] = input[0]
    total[file_completed, 1] = numpy.sum(sel)
    total[file_completed, 2] = len(sel)
    for i in numpy.argwhere(sel):
        sel_by_mic.setdefault(int(label[i, 0]), []).append(int(label[i, 1]+1))    
    output = spider_utility.spider_filename(format_utility.add_prefix(output, "avg_"), 1, id_len)
    _logger.info("Finished processing %d - %d,%d (%d,%d) - %s"%(input[0], numpy.sum(total[:file_completed+1, 1]), numpy.sum(total[:file_completed, 2]), total[file_completed, 1], total[file_completed, 2], msg))
    file_completed *= len(avg3)
    for i in xrange(len(avg3)):
        ndimage_file.write_image(output, avg3[i], file_completed+i)
    # plot first 2 eigen vectors
    return input[0]

def reconstruct3(image_file, label, align, output):
    '''
    '''
    
    # Define two subsets - here even and odd
    even = numpy.arange(0, len(align), 2, dtype=numpy.int)
    odd = numpy.arange(1, len(align), 2, dtype=numpy.int)
    iter_even = ndimage_file.iter_images(image_file, label[even])
    iter_odd = ndimage_file.iter_images(image_file, label[odd])
    vol,vol_even,vol_odd = reconstruct.reconstruct_nn4_3(iter_even, iter_odd, align[even], align[odd])
    # Write volume to file
    ndimage_file.write_image(output, vol)
    ndimage_file.write_image(format_utility.add_prefix(output, "even_"), vol_even)
    ndimage_file.write_image(format_utility.add_prefix(output, "odd_"), vol_odd)

def finalize(files, total, sel_by_mic, output, input_files, selected, align, alignheader, **extra):
    # Finalize global parameters for the script
    
    '''
    if global_rank is not None:
        idx1 = global_rank[0].ravel()
        idx2 = global_rank[1].ravel()
        idx1 = numpy.argsort(global_feat[:, 0])
        idx2 = idx1[::-1]
        idx1 = idx1[:5000]
        idx2 = idx2[:5000]
        _logger.info("Reconstructing %d selected projections"%(idx1.shape[0]))
        reconstruct3(input_files[0], global_label[idx1].copy(), global_align[idx1].copy(), format_utility.add_prefix(output, "vol_sel_"))
        _logger.info("Reconstructing %d unselected projections"%(idx2.shape[0]))
        reconstruct3(input_files[0], global_label[idx2].copy(), global_align[idx2].copy(), format_utility.add_prefix(output, "vol_unsel_"))
        _logger.info("Reconstructing - finished")
    '''
    
    format.write(output, align[selected], prefix="sel_align_", default_format=format.spiderdoc, header=alignheader)
    format.write(output, align[numpy.logical_not(selected)], prefix="unsel_align_", default_format=format.spiderdoc, header=alignheader)
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
    view_stack = ('None', 'Positive', 'Negative', 'Both', 'Single')
    group = OptionGroup(parser, "AutoPick", "Options to control reference-free particle selection",  id=__name__)
    group.add_option("", local_neighbors=0,           help="Number of neighbors for local averaging of training set before PCA")
    group.add_option("", resample=0,                  help="Number of times to resample the images")
    group.add_option("", sample_size=100.0,           help="Size of each bootstrapped sample")
    group.add_option("", resolution=15.0,             help="Filter to given resolution - requires apix to be set")
    group.add_option("", resolution_hi=0.0,           help="High-pass filter to given resolution - requires apix to be set")
    group.add_option("", use_rtsq=False,              help="Use alignment parameters to rotate projections in 2D")
    group.add_option("", neig=1.0,                    help="Number of eigen vectors to use")
    group.add_option("", disable_bispec=False,        help="Use the image data rather than the bispectrum")
    group.add_option("", bispec_window=windows,       help="Use the image data rather than the bispectrum", default=6)
    group.add_option("", bispec_biased=False,         help="Estimate biased bispectrum, default unbiased")
    group.add_option("", bispec_lag=0.314,            help="Percent of the signal to be used for lag")
    group.add_option("", bispec_mode=('Both', 'Amp', 'Phase'), help="Part of the bispectrum to use", default=0)
    group.add_option("", single_view=0,                help="Test the algorithm on a specific view")
    group.add_option("", write_view_stack=view_stack, help="Write out selected views to a stack where single means write both to a single stack", default=0)
    group.add_option("", sort_view_stack=False,       help="Sort the view stack by the first Eigen vector")
    group.add_option("", shift_data=0,                help="Number of times to shift data")
    group.add_option("", use_radon=False,             help="Use the radon transform of the image")
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-a", alignment="",   help="Input file containing alignment parameters", required_file=True, gui=dict(filetype="open"))
def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if options.bispec_lag < 1e-20: raise OptionValueError, "Invalid value for `--bispec-lag`, must be greater than zero"

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

