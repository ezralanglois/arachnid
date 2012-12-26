'''

http://deeplearning.stanford.edu/wiki/index.php/Implementing_PCA/Whitening

.. Created on Sep 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import matplotlib
matplotlib.use("Agg")

from ..core.app.program import run_hybrid_program
from ..core.image.ndplot import pylab
from ..core.image import ndimage_file, eman2_utility, analysis, ndimage_utility #, reconstruct
from ..core.metadata import spider_utility, format, format_utility, spider_params
from ..core.parallel import process_queue #, mpi_utility
from ..core.image import manifold
from arachnid.core.util import plotting #, fitting
import logging, numpy, os, scipy,itertools, sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
    
    

def batch(files, output, local_neighbors=10, **extra):
    '''
    '''
    
    _logger.info("Read alignment file")
    label, align, ref = read_alignment(files, **extra)
    if 1 == 1:
        sel = numpy.logical_or(numpy.logical_or(ref == 1, ref == 2), ref==3)
        label = label[sel]
        align = align[sel]
        ref = ref[sel]
    _logger.info("Read %d images"%len(label))
    mask = create_mask(files, **extra)
    data = ndimage_file.read_image_mat(files, label, image_transform, shared=False, mask=mask, cache_file=None, align=align, **extra)
    assert(data.shape[0] == ref.shape[0])
    
    if 1 == 0:
        tst = data-data.mean(0)
        U, d, V = scipy.linalg.svd(tst, False)
        feat = d*numpy.dot(V, tst.T).T
        t = d**2/tst.shape[0]
        t /= t.sum()
        best = (1e20, None)
        for idx in xrange(1, 100):
            val = feat[:, :idx]
            neigh = manifold.knn(val.copy(), 10)
            err = 0
            cerr = 0
            col = neigh.col.reshape((len(val), 11))
            for i in xrange(len(col)):
                cnt = numpy.sum( (ref[col[i, 0]] - ref[col[i, 1:]]) != 0 )
                err += cnt
                if cnt <= 5: cerr += 1
            if err < best[0]: best=(err, idx)
            _logger.info("PCA(%d) = %f - %f - %f"%(idx, err, float(cerr)/len(col), numpy.sum(t[:idx])))
        err, idx = best
        neigh = manifold.knn(val.copy(), 10)
        col = neigh.col.reshape((len(val), 11))
        cnt = numpy.zeros(len(col))
        for i in xrange(len(col)):
            cnt[i] = numpy.sum( (ref[col[i, 0]] - ref[col[i, 1:]]) == 0 )
        order = numpy.argsort(cnt)[::-1]
        for i, img in enumerate(ndimage_file.iter_images(files[0], label[order])):
            ndimage_file.write_image("test_stack_01.spi", img, i)
        feat2=feat[:, :idx]
        if 1 == 1:
            from sklearn.manifold import locally_linear
            feat = locally_linear.locally_linear_embedding(feat[:, :idx], local_neighbors, 3)[0]
            index=None
        else:
            feat, evals, index = manifold.diffusion_maps(feat[:, :idx].copy(), 3, local_neighbors, True)
            _logger.info("Eigen: %s"%str(evals))
        label[:, 0] = 1
        label[order.squeeze(), 1] = numpy.arange(1, len(order)+1)
        if index is not None:
            index = index.squeeze()
            label = label[index]
            ref = ref[index]
            feat2 = feat2[index]
        format.write_dataset(output, numpy.hstack((feat, feat2)), 1, label, ref, prefix="pca_")
    elif 1 == 0:
        classify_outliers(label, data, ref, output)
        
    if 1 == 0:
        ref -= ref.min()
        _logger.info("Image Size: %d"%(int(numpy.sqrt(data.shape[1]))))
        if 1 == 0:
            from ..core.learn.deep import dA
            data -= data.min(axis=0)
            data /= data.max(axis=0)
            dA.apply_dA(data)
        else:
            from ..core.learn.deep import SdA
            data -= data.min(axis=0)
            data /= data.max(axis=0)
            ha = len(data)/2 
            tr = 2*ha/3
            va = ha/3 + tr
            _logger.error("n=%d -- tr=%d -- va=%d"%(len(data), tr, va))
            SdA.apply_SdA(((data[:tr], ref[:tr]), (data[tr:va], ref[tr:va]), (data[va:], ref[va:])))
            sys.stdout.flush()
        
    if 1 == 1:
        from sklearn.manifold import locally_linear
        index=None
        for i, img in enumerate(ndimage_file.iter_images(files[0], label)):
            if align[i, 1] > 179.999: img = eman2_utility.mirror(img)
            ndimage_file.write_image("lle_stack_01.spi", img, i)
        #neigh = manifold.knn(data, local_neighbors)
        #feat = locally_linear.locally_linear_embedding(neigh, local_neighbors, 5)[0]
        feat = locally_linear.locally_linear_embedding(data, local_neighbors, 5)[0]
    else:
        _logger.info("Embedding manifold on %d data points"%len(label))
        feat, evals, index = manifold.diffusion_maps(data, 9, local_neighbors, True)
    image_size=0.4
    radius=40
    plot_embedded(feat[:, 0], feat[:, 1], "manifold", label, files[0], output, image_size, radius, ref, 400)
    _logger.info("Write embedding")
    if 1 == 1:
        label[:, 0] = 1
        label[:, 1] = numpy.arange(1, len(label)+1)
    if index is not None:
        label = label[index]
        ref = ref[index]
    format.write_dataset(output, feat, 1, label, ref)
    _logger.info("Completed")

def classify_outliers(label, data, select, output):
    '''
    '''
    
    for i in xrange(3):
        tst = data-data.mean(0)
        U, d, V = scipy.linalg.svd(tst, False)
        feat = d*numpy.dot(V, tst.T).T
        best = (1e20, None)
        for idx in xrange(1, 20):
            neigh = manifold.knn(feat[:, :idx].copy(), 10)
            err = 0
            col = neigh.col.reshape((len(feat), 11))
            for i in xrange(len(col)):
                cnt = numpy.sum( (select[col[i, 0]] - select[col[i, 1:]]) != 0 )
                err += cnt
            if err < best[0]: best=(err, idx)
        err, idx = best
        feat = feat[:, :idx]
        classes = numpy.unique(select)
        good = None
        for c in classes:
            csel = select == c
            featc = feat[csel]
            sel = analysis.robust_rejection(featc, 3)
            csel[numpy.logical_not(sel)]=0
            if good is None: good = csel
            else: good = numpy.logical_or(good, csel)
        format.write_dataset(output, feat, 1, label, select, prefix="pca_%d_"%(i+1))
        data = data[good].copy()
        select = select[good].copy()
        label = label[good].copy()
    
def create_mask(files, pixel_diameter, resolution, apix, **extra):
    '''
    '''
    
    img = ndimage_file.read_image(files[0])
    mask = eman2_utility.model_circle(int(pixel_diameter/2.0), img.shape[0], img.shape[1])
    bin_factor = max(1, min(8, resolution / (apix*2))) if resolution > (2*apix) else 1
    _logger.info("Decimation factor %f for resolution %f and pixel size %f"%(bin_factor, resolution, apix))
    if bin_factor > 1: mask = eman2_utility.decimate(mask, bin_factor)
    return mask

def image_transform(img, i, mask, resolution, apix, var_one=True, align=None, **extra):
    '''
    '''
    
    assert(align[i, 0]==0)
    if align[i, 1] > 179.999: img = eman2_utility.mirror(img)
    ndimage_utility.vst(img, img)
    bin_factor = max(1, min(8, resolution / (apix*2))) if resolution > (2*apix) else 1
    if bin_factor > 1: img = eman2_utility.decimate(img, bin_factor)
    ndimage_utility.normalize_standard(img, mask, var_one, img)
    #img = ndimage_utility.compress_image(img, mask)
    return img

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
    if test is not None:
        if local_neighbors > 0:
            train = manifold.local_neighbor_average(train, manifold.knn(train, local_neighbors))
        test = process_queue.recreate_global_dense_matrix(test)
        eigs_tst, idx, vec, energy = analysis.pca(train, test, neig, train.mean(axis=0))
        _logger.info("Test energy: %f"%energy)
    else:
        test = train.copy()
        eigs_tst = None
        if local_neighbors > 0:
            train = manifold.local_neighbor_average(train, manifold.knn(train, local_neighbors))
    try:
        eigs, idx, vec, energy = analysis.pca(train, train, neig, train.mean(axis=0))
    except:
        eigs = numpy.zeros((len(train), neig))
        sel = numpy.ones(len(train), dtype=numpy.bool)
        energy = -1.0
        idx = 0
    else:
        sel = one_class_classification(eigs, train)
    
    if 1 == 1:
        if local_neighbors > 0:
            train = manifold.local_neighbor_average(train, manifold.knn(train, local_neighbors))
        feat, evals, index = manifold.diffusion_maps(train, 2, 40, True)
        _logger.info("Eigen values: %s"%",".join([str(v) for v in evals]))
        if index is not None:
            feat_old = feat
            feat = numpy.zeros((train.shape[0], feat.shape[1]))
            feat[index] = feat_old
        if eigs_tst is not None:
            eigs = numpy.hstack((eigs, eigs_tst, eigs*eigs_tst, feat))
        else:
            eigs = numpy.hstack((eigs, feat))
    else:
        if eigs_tst is not None:
            eigs = numpy.hstack((eigs, eigs_tst, eigs*eigs_tst))
    
    return eigs, sel, (energy, idx)

def write_stack(input_files, label, align, output, subset=None, use_rtsq=False, single_view=False, **extra):
    '''
    '''
    
    if not single_view: return
    if subset is not None:
        label = label[subset]
        align = align[subset]
    
    for i, img in enumerate(ndimage_file.iter_images(input_files, label)):
        m = align[i, 1] > 179.999
        if use_rtsq: img = eman2_utility.rot_shift2D(img, align[i, 5], align[i, 6], align[i, 7], m)
        elif m:      img[:,:] = eman2_utility.mirror(img)
        ndimage_file.write_image(output, img, i)

def one_class_classification(feat, samp):
    ''' Perform one-class classification using Otsu's algorithm
    
    :Parameters:
    
    feat : array
           Feature space
    
    :Returns:
    
    sel : array
          Selected samples
    '''
    
    if 1 == 0:
        sel = analysis.robust_rejection(-feat[:, 0], 1.5)
        return sel
    
    feat = feat.copy()
    cent = numpy.median(feat, axis=0)
    dist_cent = scipy.spatial.distance.cdist(feat, cent.reshape((1, len(cent))), metric='euclidean').ravel()
    
    if 1 == 0:
        idx = numpy.argsort(dist_cent)
        window = 25
        mvar = numpy.zeros((len(idx)))
        for i in xrange(window, len(idx)-window-1):
            mvar[idx[i]] = numpy.mean(numpy.std(feat[i-window:i+window+1], axis=0))
        pylab.clf()
        pylab.plot(numpy.arange(1, len(mvar)+1), mvar)
        pylab.savefig(format_utility.new_filename("test_eigs.png", "mvar_", ext="png"))
        
        for i in xrange(window, len(idx)-window-1):
            mvar[idx[i]] = numpy.mean(numpy.std(samp[i-window:i+window+1], axis=0))
        pylab.clf()
        pylab.plot(numpy.arange(1, len(mvar)+1), mvar)
        pylab.savefig(format_utility.new_filename("test_samp.png", "mvar_", ext="png"))
        
        idx = numpy.argsort(feat[:, 0])
        template = numpy.mean(samp, axis=0)
        for i in xrange(window, len(idx)-window-1):
            mvar[idx[i]] = numpy.sum(numpy.square(numpy.mean(samp[i-window:i+window+1], axis=0)-template))
        pylab.clf()
        pylab.plot(numpy.arange(1, len(mvar)+1), mvar)
        pylab.savefig(format_utility.new_filename("test_samp0.png", "mvar_", ext="png"))
        
        idx = numpy.argsort(feat[:, feat.shape[1]-1])
        for i in xrange(window, len(idx)-window-1):
            mvar[idx[i]] = numpy.sum(numpy.square(numpy.mean(samp[i-window:i+window+1], axis=0)-template))
        pylab.clf()
        pylab.plot(numpy.arange(1, len(mvar)+1), mvar)
        pylab.savefig(format_utility.new_filename("test_samp7.png", "mvar_", ext="png"))
    
    sel = analysis.robust_rejection(dist_cent, 2.5)
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
    if len(label) > 0: avg /= (i+1)
    return avg



def image_transform2(img, idx, align, mask, hp_cutoff, use_rtsq=False, template=None, resolution=0.0, apix=None, unaligned=False, disable_bispec=False, use_radon=False, bispec_window='gaussian', bispec_biased=False, bispec_lag=0.0081, bispec_mode=0, flip_mirror=False, pixel_diameter=None, **extra):
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
    if unaligned:
        if not use_rtsq:
            img = eman2_utility.rot_shift2D(img, -align[5], 0, 0, 0)
    else:
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
        img *= mask
        try:
            img, freq = ndimage_utility.bispectrum(img, bispec_lag, bispec_window, scale) # 1.0 = apix
        except:
            _logger.error("%d, %d"%(img.shape[0], bispec_lag))
            raise
        
        idx = numpy.argwhere(numpy.logical_and(freq >= hp_cutoff, freq <= apix/resolution))
        if bispec_mode == 1: 
            img = numpy.log10(numpy.abs(img.real+1))
            if idx is not None: img = img[idx[:, numpy.newaxis], idx]
        elif bispec_mode == 2: 
            img = numpy.mod(numpy.angle(img), 2*numpy.pi) #img.imag
            if idx is not None: img = img[idx[:, numpy.newaxis], idx]
        else:
            if idx is not None: img = img[idx[:, numpy.newaxis], idx]
            if 1 == 1:
                sel = img.real < 0
                amp = numpy.log10(numpy.abs(img).real+1)
                img.imag[sel] = numpy.pi-img.imag[sel]
                pha = numpy.mod(numpy.angle(img), 2*numpy.pi)
            else:
                amp = img.real
                pha = img.imag
                assert(numpy.alltrue(numpy.isfinite(amp)))
                assert(numpy.alltrue(numpy.isfinite(pha)))
            img = numpy.hstack((amp[:, numpy.newaxis], pha[:, numpy.newaxis]))
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
    else:
        mask = extra['mask']
        resolution = extra['resolution']
        apix = extra['apix']
        bin_factor = min(8, resolution / (apix*2) ) if resolution > (apix*2) else 1.0
        if bin_factor > 1: extra['mask'] = eman2_utility.decimate(extra['mask'], bin_factor)
        average=None
        template=None
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
    radius=40
    
    
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
            try: plotting.plot_images(fig, iter_single_images, x[index], eigs[index, 0], image_size, radius)
            except: pass
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
            try: plotting.plot_images(fig, iter_single_images, eigs[index, 0], eigs[index, 1], image_size, radius)
            except: pass
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
                try: plotting.plot_images(fig, iter_single_images, eigs[index, 0], eigs[index, 1], image_size, radius)
                except: pass
    fig.savefig(format_utility.new_filename(output, "pos_", ext="png"), dpi=dpi)
    
    if eigs.shape[1] > 3:
        plot_embedded(eigs[:, 2], eigs[:, 3], "unaligned", label, filename, output, image_size, radius, ref, dpi)
    if eigs.shape[1] > 5:
        plot_embedded(eigs[:, 4], eigs[:, 5], "diff", label, filename, output, image_size, radius, ref, dpi)
    if eigs.shape[1] > 7:
        plot_embedded(eigs[:, 6], eigs[:, 7], "man", label, filename, output, image_size, radius, ref, dpi)
    
def plot_embedded(x, y, title, label, filename, output, image_size, radius, ref=None, dpi=72, select=None):
    '''
    '''
    
    xs = x[select] if select is not None else x
    ys = y[select] if select is not None else y
    fig, ax = plotting.plot_embedding(x, y, select, ref, dpi=dpi)
    vals = plotting.nonoverlapping_subset(ax, xs, ys, radius, 100)
    if len(vals) > 0:
        try:
            index = select[vals].ravel() if select is not None else vals
        except:
            _logger.error("%d-%d | %d"%(numpy.min(vals), numpy.max(vals), len(select)))
        else:
            if index.shape[0] > 1:
                iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, label[index].squeeze()))
                plotting.plot_images(fig, iter_single_images, x[index], y[index], image_size, radius)
    fig.savefig(format_utility.new_filename(output, title, ext="png"), dpi=dpi)

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
    #refs = numpy.unique(ref)
    return label, align, ref


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

