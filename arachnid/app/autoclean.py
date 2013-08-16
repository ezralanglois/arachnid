'''

http://deeplearning.stanford.edu/wiki/index.php/Implementing_PCA/Whitening

.. Created on Sep 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.app.program import run_hybrid_program
from ..core.image import ndimage_file, eman2_utility, analysis, ndimage_utility, rotate, manifold
from ..core.metadata import spider_utility, format, format_utility, spider_params
from ..core.parallel import mpi_utility, openmp
from ..core.orient import healpix, orient_utility
from arachnid.core.util import plotting #, fitting
import logging, numpy, os, scipy, itertools, scipy.cluster.vq, scipy.spatial.distance

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(input_vals, input_files, output, expected, id_len=0, max_eig=30, cache_file="", avg_file="", dm=0, **extra):#, neig=1, nstd=1.5
    '''Concatenate files and write to a single output file
        
    :Parameters:
    
    input_vals : list 
                 Tuple(view id, image labels and alignment parameters)
    input_files : list
                  List of input file stacks
    output : str
             Filename for output file
    id_len : int
             Max length of SPIDER ID
    extra : dict
            Unused key word arguments
                
    :Returns:
    
    filename : str
               Current filename
    '''
    
    _logger.info("Processing view %d"%int(input_vals[0]))
    label, align = rotational_sample(*input_vals[1:], **extra)
    mask = create_mask(input_files, **extra)
    if cache_file == "": 
        cache_file=None
    else:
        cache_file = spider_utility.spider_filename(cache_file, input_vals[0])
    
    openmp.set_thread_count(1)
    
    data = ndimage_file.read_image_mat(input_files, label, image_transform, shared=False, mask=mask, cache_file=cache_file, align=align, force_mat=True, **extra)
    _logger.info("Data: %s -- %s -- %s"%(str(data.shape), str(align.shape), str(label.shape)))
    assert(data.shape[0] == align.shape[0])
    assert(data.shape[0] == label.shape[0])
    openmp.set_thread_count(extra['thread_count'])
    tst = data-data.mean(0)
    neig=extra['neig']
    neig;
    
    
    if 1 == 0: #iterate
        try:
            U, d, V = scipy.linalg.svd(tst, False)
        except:
            _logger.exception("SVD failed to converge for view %d"%input_vals[0])
            rsel = numpy.ones(len(input_vals[1]), dtype=numpy.bool)
            return input_vals, rsel
        feat = d*numpy.dot(V, tst.T).T
        t = d**2/tst.shape[0]
        t /= t.sum()
        tc = t.cumsum()
    elif 1 == 0:
        #from sklearn import decomposition
        #decomposition.MiniBatchSparsePCA(n_components=n_components, alpha=0.8
        
        #from sklearn.cluster import DBSCAN
        #D = distance.squareform(distance.pdist(X))
        #S = 1 - (D / np.max(D))
        #db = DBSCAN(eps=0.95, min_samples=10).fit(S)
        #core_samples = db.core_sample_indices_
        #labels = db.labels_
        pass
    else:
        from sklearn.covariance import OAS
        
        if dm>0:
            dist2 = manifold.knn(data, dm, 10000)
            feat, eigv, index = manifold.diffusion_maps_dist(dist2, 2)
            if index is not None:
                feat2 = numpy.zeros((data.shape[0], feat.shape[1]), dtype=feat.dtype)
                feat2[index]=feat
                feat=feat2
        elif 1 == 1:
            eigv, feat=analysis.dhr_pca(tst, tst, extra['neig'], expected, True)
            '''
            if eigv is None:
                sel = numpy.ones(tst.shape[], numpy.bool)
            else:
                sel = outlier_rejection(feat[:, :2], 0.97)
            '''
        elif 1 == 0:
            _logger.error("pca-start")
            eigv, feat=analysis.pca_fast(tst, tst, 0.06, True)[1:]
            _logger.error("pca-stop: %d"%feat.shape[1])
            sel = outlier_rejection(feat[:, :2], 0.97)
            _logger.error("pca-start: %d -> %d (%d)"%(feat.shape[0], numpy.sum(sel), feat.shape[0]-numpy.sum(sel)))
            trn = data[sel]
            trn = trn - trn.mean(0)
            eigv, feat=analysis.pca_fast(trn, tst, 0.1, True)[1:]
            _logger.error("pca-stop: %d"%feat.shape[1])
        else:
            robust_cov = OAS().fit(tst)
            cov=robust_cov.covariance_
            d, V = numpy.linalg.eigh(cov)
            idx = d.argsort()[::-1] 
            d = d[idx]
            V = V[:,idx]
            feat = numpy.dot(V, tst.T).T
        
        _logger.info("Eigen: %s"%(",".join([str(v) for v in eigv[:10]])))
        if eigv is not None:
            d=eigv
            t=d
            tc=d.cumsum()
    
        
        _logger.info("Eigen: %s"%(",".join([str(v) for v in t[:10]])))
        _logger.info("Eigen-cum: %s"%(",".join([str(v) for v in tc[:10]])))
        rsel=None
        if feat is not None:
            sel, rsel, dist = one_class_classification(feat, **extra)
            try:
                format.write_dataset(output, numpy.hstack((sel[:, numpy.newaxis], dist[:, numpy.newaxis], align[:, 0][:, numpy.newaxis], label[:, 1][:, numpy.newaxis], align[:, (5,6,7,1)], feat)), input_vals[0], label, header='select,dist,rot,group,psi,tx,ty,mirror', prefix='pca_')
            except:
                _logger.error("sel: %s - dist: %s - align: %s - label: %s"%(str(sel.shape), str(dist.shape), str(align[:, 0].shape), str(label[:, 1].shape)))
                raise
            _logger.info("Finished embedding view: %d"%int(input_vals[0]))
        else:
            _logger.info("Skipping view (too few projections): %d"%int(input_vals[0]))
    
    if avg_file != "":
        avg_file = spider_utility.spider_filename(avg_file, input_vals[0])
        avg = None
        for i, img in enumerate(ndimage_file.iter_images(input_files[0], label)):
            if extra['use_rtsq']: img = rotate.rotate_image(img, align[i, 5], align[i, 6], align[i, 7])
            if avg is None: avg = img
            else: avg += img
        ndimage_file.write_image(avg_file, avg)
    
    return input_vals, rsel

def outlier_rejection(feat, prob):
    '''
    '''
    
    from sklearn.covariance import MinCovDet
    import scipy.stats, scipy.spatial.distance
    
    #real_cov
    #linalg.inv(real_cov)
    
    robust_cov = MinCovDet().fit(feat)
    dist = robust_cov.mahalanobis(feat - numpy.median(feat, 0))
    
    if 1 == 0:
        cov = analysis.dhr_cov(feat - numpy.median(feat, 0))
        cent = numpy.median(feat, 0)
        dist = scipy.spatial.distance.mahalanobis(feat, cent, cov)
    
    cut = scipy.stats.chi2.ppf(prob, feat.shape[1])
    return dist < cut

def one_class_classification(feat, neig, nsamples, prob_reject, nstd=2.5, **extra):
    '''
    '''
    
    if 1 == 1:
        from sklearn.covariance import MinCovDet,EmpiricalCovariance
        import scipy.stats
        feat=feat[:, :neig]
        try:
            robust_cov = MinCovDet().fit(feat)
            #robust_cov = GraphLasso().fit(feat)
            #robust_cov.location_ = feat.median(0)
        except:
            robust_cov = EmpiricalCovariance().fit(feat)
        dist = robust_cov.mahalanobis(feat - numpy.median(feat, 0))
            
        #dist = robust_cov.mahalanobis(feat)
        cut = scipy.stats.chi2.ppf(prob_reject, feat.shape[1])
        _logger.info("Cutoff: %d -- for df: %d"%(cut, feat.shape[1]))
        sel = dist < cut
        #sel = analysis.robust_rejection(dist, nstd*1.4826)
    elif 1 == 1:
        from sklearn.covariance import EllipticEnvelope
        import scipy.stats
        feat=feat[:, :neig]
        robust_cov = EllipticEnvelope(contamination=0.25).fit(feat)
        #dist = robust_cov.mahalanobis(feat - numpy.mean(feat, 0))
        dist = robust_cov.decision_function(feat)
        #sel = analysis.robust_rejection(-dist, nstd*1.4826)
        threshold = scipy.stats.scoreatpercentile(dist, 100 * 0.25)
        sel = dist > threshold
        _logger.info("Threshold=%f -- %d -> %d | %d"%(threshold, dist.shape[0], numpy.sum(sel), nsamples))
        #sel = analysis.robust_rejection(dist, nstd*1.4826)
    elif 1 == 1: 
        feat = feat - feat.min(axis=1).reshape((feat.shape[0], 1))
        feat /= feat.max(axis=1).reshape((feat.shape[0], 1))
        cent = numpy.median(feat[:, :neig], axis=0)
        dist_cent = scipy.spatial.distance.cdist(feat[:, :neig], cent.reshape((1, len(cent))), metric='euclidean').ravel()
        sel = analysis.robust_rejection(numpy.abs(dist_cent), nstd*1.4826)
    else:
        sel=None
        for i in xrange(neig):
            if 1 == 1:
                sel1 = analysis.robust_rejection(numpy.abs(feat[:, i]), nstd*1.4826)
            else:
                sel1 = analysis.robust_rejection(feat[:, i], nstd)
                sel2 = analysis.robust_rejection(-feat[:, i], nstd)
                if numpy.sum(sel1) > numpy.sum(sel2): sel1=sel2
            sel = numpy.logical_and(sel1, sel) if sel is not None else sel1
    if nsamples > 1:
        rsel = numpy.ones(int(feat.shape[0]/nsamples), dtype=numpy.bool)
        for i in xrange(rsel.shape[0]):
            rsel[i] = numpy.alltrue(sel[i*nsamples:(i+1)*nsamples])
    else: rsel = sel
    return sel, rsel, dist

def init_root(files, param):
    # Initialize global parameters for the script
    
    if param['order'] < 0:
        param['order'] = order_from_resolution(**param)
    
    if mpi_utility.is_root(**param):
        if param['resolution'] > 0.0: _logger.info("Filter and decimate to resolution %f by a factor of %f"%(resolution_from_order(**param), decimation_level(**param)))
        if param['use_rtsq']: _logger.info("Rotate and translate data stack")
        _logger.info("Rejection precision: %f"%param['prob_reject'])
        _logger.info("Number of Eigenvalues: %f"%param['neig'])
        if param['order'] > 0: _logger.info("Angular order %f sampling %f degrees "%(param['order'], healpix.nside2pixarea(param['order'], True)))
        #_logger.info("nsamples: %f"%param['nsamples'])
        
    group = None
    if mpi_utility.is_root(**param): 
        param['sel_by_mic']={}
        group = group_by_reference(*read_alignment(files, **param))
        if param['output_embed'] == "": param['output_embed'] = format_utility.add_prefix(param['output'], "pca_") #spider_utility.add_spider_id(format_utility.add_prefix(param['output'], "pca_"), len(str(len(group))))
        if param['single_view'] > 0:
            _logger.info("Using single view: %d"%param['single_view'])
            tmp=group
            group = [tmp[param['single_view']-1]]
        else:
            count = numpy.zeros((len(group)))
            for i in xrange(count.shape[0]):
                count[i] = len(group[i][1])
            index = numpy.argsort(count)
            newgroup=[]
            for i in index[::-1]:
                if count[i] > 20:
                    newgroup.append(group[i])
            group=newgroup
    group = mpi_utility.broadcast(group, **param)
    _logger.info("Processing %d groups"%len(group))
    return group

def reduce_all(val, sel_by_mic, output, id_len=0, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    _logger.info("Reducing to root selections")
    input, sel = val
    label = input[1]
    for i in numpy.argwhere(sel):
        sel_by_mic.setdefault(int(label[i, 0]), []).append(int(label[i, 1]+1))    
    
    tot=numpy.sum(sel)
    return "%d - Selected: %d -- Removed %d"%(input[0], tot, label.shape[0]-tot)

def finalize(files, output, output_embed, sel_by_mic, finished, nsamples, thread_count, neig, input_files, **extra):
    # Finalize global parameters for the script
    
    nsamples = None
            
    for filename in finished:
        label = filename[1]
        data = format.read(output_embed, numeric=True, spiderid=int(filename[0]))
        feat, header = format_utility.tuple2numpy(data)
        if nsamples is None:
            nsamples = len(numpy.unique(feat[:, 4]))
            _logger.info("Number of samples per view: %d"%nsamples)
        feat = feat[:, 6:]
        sel, rsel = one_class_classification(feat, neig=neig, nsamples=nsamples, **extra)[:2]
        _logger.info("Read %d samples and selected %d from finished view: %d"%(feat.shape[0], numpy.sum(rsel), int(filename[0])))
        for j in xrange(len(feat)):
            data[j] = data[j]._replace(select=sel[j])
        format.write(output, data, prefix='pca_', spiderid=int(filename[0]))
        #sel = [val.select for val in format.read(output, numeric=True, spiderid=int(filename[0]))]
        for i in numpy.argwhere(rsel):
            sel_by_mic.setdefault(int(label[i, 0]), []).append(int(label[i, 1]+1))
    tot=0
    for id, sel in sel_by_mic.iteritems():
        n=len(sel)
        tot+=n
        _logger.info("Writing %d to selection file %d"%(n, id))
        sel = numpy.asarray(sel)
        format.write(output, numpy.vstack((sel, numpy.ones(sel.shape[0]))).T, prefix="sel_", spiderid=id, header=['id', 'select'], default_format=format.spidersel)
    _logger.info("Selected %d projections"%(tot))
    _logger.info("Completed")

def rotational_sample(label, align, nsamples, angle_range, **extra):
    '''
    '''
    
    if nsamples < 2:
        return label, align
    label2 = numpy.zeros((label.shape[0]*nsamples, label.shape[1]))
    align2 = numpy.zeros((align.shape[0]*nsamples, align.shape[1]))
    for i in xrange(len(label)):
        label2[i*nsamples:(i+1)*nsamples] = label[i]
        align2[i*nsamples:(i+1)*nsamples] = align[i]
        align2[i*nsamples:(i+1)*nsamples, 0]=scipy.linspace(-angle_range/2.0, angle_range/2.0, nsamples,True)
    return label2, align2

def create_mask(files, pixel_diameter, apix, **extra):
    '''
    '''
    
    img = ndimage_file.read_image(files[0])
    mask = eman2_utility.model_circle(int(pixel_diameter/2.0), img.shape[0], img.shape[1])
    bin_factor = decimation_level(apix, pixel_diameter=pixel_diameter, **extra)
    #bin_factor = max(1, min(8, resolution / (apix*4))) if resolution > (4*apix) else 1
    #_logger.info("Decimation factor %f for resolution %f and pixel size %f"%(bin_factor,  resolution, apix))
    if bin_factor > 1: mask = eman2_utility.decimate(mask, bin_factor)
    return mask

def resolution_from_order(apix, pixel_diameter, order, resolution, **extra):
    '''
    '''
    
    if order == 0: return resolution
    res = numpy.tan(healpix.nside2pixarea(order))*pixel_diameter*apix
    if res > resolution: resolution=res
    return resolution

def order_from_resolution(apix, pixel_diameter, order, resolution, **extra):
    '''
    '''
    
    theta_delta = numpy.rad2deg( numpy.arctan( resolution / (pixel_diameter*apix) ) )
    _logger.info("Target sampling %f for resolution %f -> %d"%(theta_delta, resolution, healpix.theta2nside(numpy.deg2rad(theta_delta))))
    return healpix.theta2nside(numpy.deg2rad(theta_delta))

def decimation_level(apix, window, **extra):
    '''
    '''
    
    resolution = resolution_from_order(apix, **extra)
    dec = resolution / (apix*3)
    d = float(window)/dec + 10
    d = window/float(d)
    return min(max(d, 1), 8)

def image_transform(img, i, mask, resolution, apix, var_one=True, align=None, disable_bispec=False, use_rtsq=False, **extra):
    '''
    '''
    
    if use_rtsq: img = rotate.rotate_image(img, align[i, 5], align[i, 6], align[i, 7])
    elif align[i, 0] != 0: img = rotate.rotate_image(img, align[i, 0])
    if align[i, 1] > 179.999: img = eman2_utility.mirror(img)
    ndimage_utility.vst(img, img)
    bin_factor = decimation_level(apix, resolution=resolution, **extra)
    #bin_factor = max(1, min(8, resolution / (apix*4))) if resolution > (4*apix) else 1
    if bin_factor > 1: img = eman2_utility.decimate(img, bin_factor)
    if mask.shape[0] != img.shape[0]:
        _logger.error("mask-image: %d != %d"%(mask.shape[0],img.shape[0]))
    assert(mask.shape[0]==img.shape[0])
    #ndimage_utility.normalize_standard(img, mask, var_one, img)
    ndimage_utility.normalize_standard_norm(img, mask, var_one, out=img)
    
    if not disable_bispec:
        #img *= mask
        #img = ndimage_utility.polar(img)
        img, freq = ndimage_utility.bispectrum(img, int(img.shape[0]-1), 'uniform')#gaussian
        #bispectrum(signal, maxlag=0.0081, window='gaussian', scale='unbiased')
        freq;
        img = numpy.log10(numpy.abs(img.real)+1)
        #img = numpy.angle(numpy.square(img/numpy.abs(img))) #numpy.arctan().real
    
    #img = ndimage_utility.compress_image(img, mask)
    return img
    
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

def group_by_reference(label, align, ref):
    '''
    '''
    
    group=[]
    refs = numpy.unique(ref)
    _logger.info("Processing %d projections from %d stacks grouped into %d views"%(len(label), len(numpy.unique(label[:, 0])), len(refs)))
    for r in refs:
        sel = r == ref
        group.append((r, label[sel], align[sel]))
    return group

def read_alignment(files, alignment="", order=0, **extra):
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
        ididx = header.index('id')
        label = numpy.zeros((len(align), 2), dtype=numpy.int)
        label[:, 0] = spider_utility.spider_id(files[0])
        label[:, 1] = align[:, ididx].astype(numpy.int)-1
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
    if order > 0:
        resolution = pow(2, order)
        ang = healpix.angles(order)
        for i in xrange(len(align)):
            t = align[i, 1]
            if t > 180.0: t -= 180.0
            align[i, refidx] = healpix._healpix.ang2pix_ring(resolution, numpy.deg2rad(t), numpy.deg2rad(align[i, 2]))
            rang = rotate.rotate_euler(ang[int(align[i, refidx])], (-align[i, 5], align[i, 1], align[i, 2]))
            rot = (rang[0]+rang[2])
            rt3d = orient_utility.align_param_2D_to_3D_simple(align[i, 5], align[i, 6], align[i, 7])
            align[i, 5], align[i, 6], align[i, 7]=orient_utility.align_param_2D_to_3D_simple(rot, rt3d[1], rt3d[2])
            
            
    ref = align[:, refidx].astype(numpy.int)
    #refs = numpy.unique(ref)
    return label, align, ref

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "AutoPick", "Options to control reference-free particle selection",  id=__name__)
    group.add_option("", nsamples=1,                 help="Number of rotational samples")
    group.add_option("", angle_range=3.0,              help="Angular search range")
    group.add_option("", resolution=40.0,             help="Filter to given resolution - requires apix to be set")
    group.add_option("", use_rtsq=False,              help="Use alignment parameters to rotate projections in 2D")
    group.add_option("", neig=2,                    help="Number of eigen vectors to use", dependent=False)
    group.add_option("", expected=0.8,                    help="Expected fraction of good data", dependent=False)
    group.add_option("", max_eig=8,                    help="Maximum number of eigen vectors saved")
    #group.add_option("", nstd=2.5,                    help="Number of deviations from the median", dependent=False)
    group.add_option("", single_view=0,                help="Test the algorithm on a specific view")
    group.add_option("", disable_bispec=False,          help="Disable bispectrum feature space")
    group.add_option("", cache_file="",                 help="Cache preprocessed data in matlab data files")
    group.add_option("", order=0,                      help="Reorganize views based on their healpix order (overrides the resolution parameter)")
    group.add_option("", prob_reject=0.97,             help="Probablity that a rejected particle is bad", dependent=False)
    group.add_option("", avg_file="",                   help="Write out an average for each view")
    group.add_option("", dm=0,                         help="Use experimental mode")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with with no digits at the end (e.g. this is bad -> sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("",   output_embed="", help="Output filename for the coordinate file with correct number of digits (e.g. sndc_0000.spi) - this does not need to be set", gui=dict(filetype="save"), required_file=False)
        pgroup.add_option("-a", alignment="",   help="Input file containing alignment parameters", required_file=True, gui=dict(filetype="open"))

def check_options(options, main_option=False):
    #Check if the option values are valid
#    from ..core.app.settings import OptionValueError

    if options.nsamples < 1: options.nsamples=1

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

