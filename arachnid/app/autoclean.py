'''
.. Created on Sep 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.image.ndplot import pylab
from ..core.image import ndimage_file, eman2_utility, analysis, ndimage_utility
from ..core.metadata import spider_utility, format, format_utility, spider_params
from ..core.parallel import mpi_utility, process_queue
from arachnid.core.util import plotting, fitting
import logging, numpy, os, scipy,itertools

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
    data, mask, avg, template = read_data(input_files, *input_vals[1:3], **extra)
    #data2 = data
    #data2, mask = read_data(input_files, *input_vals[3:5], mask=mask, **extra)
    #_logger.error("test: %s == %s"%(str(data.shape), str(data2.shape)))
    #data2 = numpy.vstack((data, data2))
    #_logger.error("test2: %s == %s"%(str(data.shape), str(data2.shape)))
    eigs, sel, energy = classify_data(data, None, output=output, **extra)
    test_covariance(eigs, data, output)
    #sel = numpy.zeros((len(data2)))
    #sel[:len(data)]=1
    
    # Write 
    if eigs.ndim == 1: eigs = eigs.reshape((eigs.shape[0], 1))
    feat = numpy.hstack((sel[:, numpy.newaxis], eigs))
    assert(feat.ndim > 1)
    #header=[]
    #for i in xrange(1, eigs.shape[1]+1): header.append('c%d'%i)
    nexamples = data.shape[0] if hasattr(data, 'shape') else data[1][0]
    label = numpy.zeros((nexamples, 2))
    label[:, 0] = input_vals[0]
    label[:, 1] = numpy.arange(1, nexamples+1)
    format.write_dataset(os.path.splitext(output)[0]+".csv", feat, input_vals[0], label, prefix="embed_", header="best")
    #format.write(os.path.splitext(output)[0]+".csv", feat, prefix="embed_", spiderid=input_vals[0], header=['select']+header)
    
    plot_examples(input_files, input_vals[1], output, eigs, sel, **extra)
    extra['poutput']=format_utility.new_filename(output, 'pos_') if write_view_stack == 1 or write_view_stack >= 3 else None
    extra['noutput']=format_utility.new_filename(output, 'neg_') if write_view_stack == 2 or write_view_stack == 3 else None
    if write_view_stack == 4: extra['noutput'] = extra['poutput']
    extra['sort_by']=eigs[:, 0] if sort_view_stack else None
    avg3 = compute_average3(input_files, *input_vals[1:3], mask=mask, selected=sel, output=output, **extra)
    extra['poutput']=None
    extra['noutput']=None
    savg3 = compute_average3(input_files, *input_vals[1:3], mask=mask, selected=sel, template=template, output=output, **extra)
    
    nfeat = data.shape[1] if hasattr(data, 'shape') else data[1][1]
    return input_vals, sel, avg3[:3]+savg3[:3], energy, nfeat, numpy.min(eigs), numpy.max(eigs) #avg3[3], savg3[3]

def classify_data(data, test=None, neig=1, thread_count=1, resample=0, sample_size=0, local_neighbors=0, **extra):
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
            from ..core.image import manifold
            test = process_queue.recreate_global_dense_matrix(data)
            train = manifold.local_neighbor_average(test, manifold.knn(samp, local_neighbors))
        else: 
            train = process_queue.recreate_global_dense_matrix(data)
            test = train
        if sel is not None: train = train[sel]
        
        if 1 == 0:
            from ..core.image import manifold
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
        eigs2 -= eigs2.min(axis=0)
        eigs2 /= eigs2.max(axis=0)
        cent = numpy.median(eigs2, axis=0)
        eig_dist_cent = scipy.spatial.distance.cdist(eigs2, cent.reshape((1, len(cent))), metric='euclidean').ravel()
        th = analysis.otsu(eig_dist_cent)
        sel = eig_dist_cent < th
        if 1 == 0:
            from ..core.image import manifold
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
    feat -= feat.min(axis=0)
    feat /= feat.max(axis=0)
    cent = numpy.median(feat, axis=0)
    dist_cent = scipy.spatial.distance.cdist(feat, cent.reshape((1, len(cent))), metric='euclidean').ravel()
    return dist_cent < analysis.otsu(dist_cent)

def image_transform(img, idx, align, mask, hp_cutoff, use_rtsq=False, template=None, resolution=0.0, apix=None, disable_bispec=False, bispec_window='gaussian', bispec_biased=False, bispec_lag=0.0081, bispec_mode=0, flip_mirror=False, pixel_diameter=None, **extra):
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
    if template is not None: img = shift(img, template, pixel_diameter/2)
    if not disable_bispec:
        #ndimage_utility.normalize_standard(img, mask*-1+1, var_one, img)
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
            img = img.imag
            img = img[idx[:, numpy.newaxis], idx]
        else:
            img = img[idx[:, numpy.newaxis], idx]
            img = numpy.hstack((img.real.ravel()[:, numpy.newaxis], img.imag.ravel()[:, numpy.newaxis]))
        ndimage_utility.normalize_standard(img, None, var_one, img)
    else:
        ndimage_utility.normalize_standard(img, mask, var_one, img)
        img = ndimage_utility.compress_image(img, mask)
    return img

def shift(img, template, rad, shiftval=None):
    ''' Shift an image to match the given template
    
    :Parameters:
    
    img : array
          Image to shift
    rad : pixel_radius
           Search radius
    template : array
               Image to match
    
    :Returns:
    
    img : array
          Shifted image
    '''
    
    #cc_map = ndimage_utility.cross_correlate(img, template)
    cc_map = ndimage_utility.cross_correlate(template, img)
    y,x = numpy.unravel_index(numpy.argmax(cc_map*eman2_utility.model_circle(rad, cc_map.shape[0], cc_map.shape[1])), cc_map.shape)
    if shiftval is not None:
        shiftval[:] = (x-img.shape[0]/2,y-img.shape[1]/2)
    return eman2_utility.fshift(img, x-img.shape[0]/2, y-img.shape[1]/2)

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
    return ndimage_file.read_image_mat(input_files, label, image_transform, shared=True, align=align, **extra), mask, average, template

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
    
    rad = pixel_diameter/2
    avg2 = None
    for img2 in ndimage_file.iter_images(input_files, label):
        if avg2 is None: avg2 = numpy.zeros(img2.shape)
        img2 = shift(img2, template, rad)
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
    ref = align[:, refidx].astype(numpy.int)
    refs = numpy.unique(ref)
    
    group = []
    for r in refs:
        sel = r == ref
        #sel2 = refs[0] == ref if r != refs[0] else refs[1] == ref
        group.append((r, label[sel], align[sel]))#, label[sel2], align[sel2]))
    return group, align.shape[0]

def compute_average3(input_files, label, align, template=None, selected=None, mask=None, use_rtsq=False, sort_by=None, noutput=None, poutput=None, pixel_diameter=None, output=None, **extra):
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
    pixel_diameter : int
                     Diameter of particle in pixels
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
    
    rad = pixel_diameter/2
    shiftval = numpy.zeros((label.shape[0], 2)) if template is not None else None
    avgsel, avgunsel = None, None
    cntsel  = numpy.sum(selected)
    cntunsel = selected.shape[0] - cntsel
    poff = 0
    noff = 0
    for i, img in enumerate(ndimage_file.iter_images(input_files, label)):
        if avgsel is None: avgsel = numpy.zeros((2, )+img.shape)
        if avgunsel is None: avgunsel = numpy.zeros((2, )+img.shape)
        m = align[i, 1] > 179.999
        if use_rtsq: img = eman2_utility.rot_shift2D(img, align[i, 5], align[i, 6], align[i, 7], m)
        elif m:      img[:,:] = eman2_utility.mirror(img)
        if template is not None: img = shift(img, template, rad, shiftval)
        idx = 1 if i%2==0 else 0
        if selected[i] > 0.5:
            avgsel[idx] += img
            if poutput is not None and poutput != noutput:
                ndimage_file.write_image(poutput, img, poff)
                poff += 1
        else:
            avgunsel[idx] += img
            if noutput is not None and poutput != noutput:
                ndimage_file.write_image(noutput, img, noff)
                noff += 1
        if poutput is not None and poutput == noutput:
            ndimage_file.write_image(poutput, img, poff)
            poff += 1
    if avgunsel is None: avgunsel = numpy.zeros(img.shape)
    if avgsel is None: avgsel = numpy.zeros(img.shape)
    avgful = avgsel + avgunsel
    avgsel2 = (avgsel[0]+avgsel[1])
    avgunsel2 = (avgunsel[0]+avgunsel[1])
    if numpy.sum(cntsel) > 0: avgsel2 /= numpy.sum(cntsel)
    if numpy.sum(cntunsel) > 0: avgunsel2 /= numpy.sum(cntunsel)
    avgful2 = avgsel2 + avgunsel2
    if cntsel > 0: avgsel /= (cntsel/2)
    if cntunsel > 0: avgunsel /= (cntunsel/2)
    
    if 1 == 1:
        #fsc_all = extra['apix']/fitting.fit_linear_interp(ndimage_utility.frc(avgful[0], avgful[1]), 0.5)
        #res = 
        #fsc_all = extra['apix']/fitting.fit_sigmoid_interp(eman2_utility.fsc(avgful[0], avgful[1], complex=False), 0.5)
        #fsc = ndimage_utility.frc(avgful[0]*mask, avgful[1]*mask, pad=1)
        
        #pylab.clf()
        #pylab.plot(fsc[:, 0], fsc[:, 1], 'r-')
        #pylab.savefig(format_utility.new_filename(output, "fsc_", ext="png"))
        
        fsc1 = eman2_utility.fsc(avgful[0]*mask, avgful[1]*mask)
        fsc2 = eman2_utility.fsc(avgsel[0]*mask, avgsel[1]*mask)
        fsc3 = eman2_utility.fsc(avgunsel[0]*mask, avgunsel[1]*mask)
        fsc1 = extra['apix']/fitting.fit_linear_interp(fsc1, 0.5)
        fsc2 = extra['apix']/fitting.fit_linear_interp(fsc2, 0.5)
        fsc3 = extra['apix']/fitting.fit_linear_interp(fsc3, 0.5)
        '''
        _logger.info("Compare: %f -> %f, %f"%())
        try:
            numpy.testing.assert_allclose(fsc2[:, 0], fsc[:, 0])
            numpy.testing.assert_allclose(fsc2[:, 1], fsc[:, 1])
        except:
            _logger.exception("Did not work")
        '''
        #_logger.info("fsc: %f, %f ... %f,%f,%f"%(fsc[0, 1], fsc[1, 1], fsc[len(fsc)-3, 1], fsc[len(fsc)-2, 1], fsc[len(fsc)-1, 1]))
        #fsc_all = extra['apix']/fitting.fit_sigmoid_interp(fsc, 0.5)
        #if fsc_all < 0 or fsc_all > 1000:
        fsc_all = fsc1#extra['apix']/fitting.fit_linear_interp(fsc, 0.5)
        #fsc_sel = fitting.fit_linear_interp(ndimage_utility.frc(avgsel[0], avgsel[1]), 0.5)
        #fsc_uns = fitting.fit_linear_interp(ndimage_utility.frc(avgunsel[0], avgunsel[1]), 0.5)
    else:
        fsc_all = numpy.sum(numpy.abs(shiftval))/float(label.shape[0]) if shiftval is not None else 0
    return avgsel2, avgunsel2, avgful2 / (cntsel+cntunsel), fsc_all

def test_covariance(eigs, data, output, **extra):
    '''
    '''
    
    
    data = process_queue.recreate_global_dense_matrix(data)
    cent = numpy.median(eigs, axis=0)
    eig_dist_cent = scipy.spatial.distance.cdist(eigs, cent.reshape((1, len(cent))), metric='euclidean').ravel()
    idx = numpy.argsort(eig_dist_cent)
    
    
    dcov = numpy.cov(data)
    ecov = numpy.cov(eigs)
    dmcov = numpy.zeros(idx.shape[0])
    emcov = numpy.zeros(idx.shape[0])
    for i in xrange(1, idx.shape[0]):
        dmcov[i] = numpy.mean(dcov[idx, idx[:, numpy.newaxis]])
        emcov[i] = numpy.mean(ecov[idx, idx[:, numpy.newaxis]])
    x = numpy.arange(1, len(ecov)+1)
    pylab.clf()
    pylab.plot(x, ecov, 'r+.', x, emcov, 'g*.')
    pylab.savefig(format_utility.new_filename(output, "cov_", ext="png"))

def test_variance(eigs, data, output, **extra):
    '''
    '''
    
    cent = numpy.median(eigs, axis=0)
    eig_dist_cent = scipy.spatial.distance.cdist(eigs, cent.reshape((1, len(cent))), metric='euclidean').ravel()
    idx = numpy.argsort(eig_dist_cent)
    var = analysis.online_variance(data[idx], axis=0)
    rvar = analysis.online_variance(data[idx[::-1]], axis=0)
    mvar = numpy.mean(var, axis=1)
    mrvar = numpy.mean(rvar, axis=1)
    pylab.clf()
    pylab.plot(numpy.arange(1, len(mvar)+1), mvar)
    pylab.savefig(format_utility.new_filename(output, "mvar_", ext="png"))
    pylab.clf()
    pylab.plot(numpy.arange(1, len(mrvar)+1), mrvar)
    pylab.savefig(format_utility.new_filename(output, "mrvar_", ext="png"))
    
    th = analysis.otsu(eig_dist_cent)
    pylab.clf()
    n = pylab.hist(eig_dist_cent, bins=int(numpy.sqrt(len(mvar))))[0]
    maxval = sorted(n)[-1]
    pylab.plot((th, th), (0, maxval))
    pylab.savefig(format_utility.new_filename(output, "den_", ext="png"))
    
def plot_examples(filename, label, output, eigs, sel, dpi=200, **extra):
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
    fig, ax = plotting.plot_embedding(eigs[:, 0], eigs[:, 1], select, dpi=dpi)
    index = select[plotting.nonoverlapping_subset(ax, eigs[select, 0], eigs[select, 1], radius, 100)].ravel()
    #_logger.error("selected: %s"%str(index.shape))
    iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, label[index].squeeze()))
    plotting.plot_images(fig, iter_single_images, eigs[index, 0], eigs[index, 1], image_size, radius)
    fig.savefig(format_utility.new_filename(output, "neg_", ext="png"), dpi=dpi)
    
    select = numpy.argwhere(sel > 0.5)
    fig, ax = plotting.plot_embedding(eigs[:, 0], eigs[:, 1], select, dpi=dpi)
    index = select[plotting.nonoverlapping_subset(ax, eigs[select, 0], eigs[select, 1], radius, 100)].ravel()
    if index.shape[0] > 1:
        iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, label[index].squeeze()))
        plotting.plot_images(fig, iter_single_images, eigs[index, 0], eigs[index, 1], image_size, radius)
    fig.savefig(format_utility.new_filename(output, "pos_", ext="png"), dpi=dpi)
    
''''''
def initialize(files, param):
    # Initialize global parameters for the script
    
    if mpi_utility.is_root(**param):
        if param['resolution'] > 0.0: _logger.info("Filter and decimate to resolution: %f"%param['resolution'])
        if param['use_rtsq']: _logger.info("Rotate and translate data stack")
    
    group = None
    if mpi_utility.is_root(**param): 
        group, total = read_alignment(files, **param)
        _logger.info("Cleaning bad particles from %d views"%len(group))
    group = mpi_utility.broadcast(group, **param)
    if param['single_view'] > 0:
        tmp=group
        group = [tmp[param['single_view']-1]]
    param['total'] = numpy.zeros((len(group), 3))
    param['sel_by_mic'] = {}
    
    return group

def reduce_all(val, input_files, total, sel_by_mic, output, file_completed, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    input, sel, avg3, energy, feat_cnt, ndiff, sdiff = val
    label = input[1]
    file_completed -= 1
    total[file_completed, 0] = input[0]
    total[file_completed, 1] = numpy.sum(sel)
    total[file_completed, 2] = len(sel)
    for i in numpy.argwhere(sel):
        sel_by_mic.setdefault(int(label[i, 0]), []).append(int(label[i, 1]+1))    
    output = spider_utility.spider_filename(format_utility.add_prefix(output, "avg_"), 1)
    _logger.info("Finished processing %d - %d,%d (%d,%d) - Energy: %f, %d - Features: %d - Range: %f,%f"%(input[0], numpy.sum(total[:file_completed+1, 1]), numpy.sum(total[:file_completed, 2]), total[file_completed, 1], total[file_completed, 2], energy[0], energy[1], feat_cnt, ndiff, sdiff))
    file_completed *= len(avg3)
    for i in xrange(len(avg3)):
        ndimage_file.write_image(output, avg3[i], file_completed+i)
    # plot first 2 eigen vectors
    return input[0]

def finalize(files, total, sel_by_mic, output, **extra):
    # Finalize global parameters for the script
    
    format.write(output, total, prefix="sel_avg_", header=['id', 'selected', 'total'], default_format=format.spiderdoc)
    for id, sel in sel_by_mic.iteritems():
        sel = numpy.asarray(sel)
        format.write(output, numpy.vstack((sel, numpy.ones(sel.shape[0]))).T, prefix="sel_", spiderid=id, header=['id', 'select'], default_format=format.spidersel)
    _logger.info("Selected %d out of %d"%(numpy.sum(total[:, 1]), numpy.sum(total[:, 2])))
    _logger.info("Completed")

'''
def initialize(files, param):
    # Initialize global parameters for the script
    
    if mpi_utility.is_root(**param):
        if param['resolution'] > 0.0: _logger.info("Filter and decimate to resolution: %f"%param['resolution'])
        if param['use_rtsq']: _logger.info("Rotate and translate data stack")
    
    group = None
    total=0
    if mpi_utility.is_root(**param): 
        group, total = read_alignment(files, **param)
        _logger.info("Cleaning bad particles from %d views"%len(group))
        param['global_label'] = numpy.zeros((total, 2))
        assert(extra['neig']>=1)
        param['global_feat'] = numpy.zeros((total, extra['neig']))
        param['global_offset'] = numpy.zeros((1))
    group = mpi_utility.broadcast(group, **param)
    if param['single_view'] > 0:
        tmp=group
        group = [tmp[param['single_view']-1]]
    
    return group

def reduce_all(val, input_files, global_label, global_feat, global_offset, output, file_completed, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    input, sel, avg3, eigs, label = val
    label = input[1]
    file_completed -= 1
    global_label[global_offset[0]:global_offset[0]+len(label)] = label
    global_feat[global_offset[0]:global_offset[0]+len(eigs)] = eigs
    global_offset[0]+=len(label)

    output = spider_utility.spider_filename(format_utility.add_prefix(output, "avg_"), 1)
    _logger.info("Finished processing %d"%(input[0]))
    file_completed *= len(avg3)
    for i in xrange(len(avg3)):
        ndimage_file.write_image(output, avg3[i], file_completed+i)
    # plot first 2 eigen vectors
    return input[0]

def finalize(files, global_label, global_feat, output, **extra):
    # Finalize global parameters for the script
    
    eigs2 = eigs.copy()
    eigs2 -= eigs2.min(axis=0)
    eigs2 /= eigs2.max(axis=0)
    cent = numpy.median(eigs2, axis=0)
    eig_dist_cent = scipy.spatial.distance.cdist(eigs2, cent.reshape((1, len(cent))), metric='euclidean').ravel()
    th = analysis.otsu(eig_dist_cent)
    sel = eig_dist_cent < th
    
    if 1 == 0:
        format.write(output, total, prefix="sel_avg_", header=['id', 'selected', 'total'], default_format=format.spiderdoc)
        for id, sel in sel_by_mic.iteritems():
            sel = numpy.asarray(sel)
            format.write(output, numpy.vstack((sel, numpy.ones(sel.shape[0]))).T, prefix="sel_", spiderid=id, header=['id', 'select'], default_format=format.spidersel)
        _logger.info("Selected %d out of %d"%(numpy.sum(total[:, 1]), numpy.sum(total[:, 2])))
    _logger.info("Completed")
'''

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

