''' Orientation assigment


.. Created on Apr 1, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.app import program
from ..core.image import ndimage_file, eman2_utility, ndimage_utility, manifold, rotate
from ..core.metadata import spider_utility, format, spider_params, format_utility
from ..core.orient import orient_utility, healpix
from ..core.parallel import process_queue
import logging, numpy, scipy.io, os, scipy.sparse
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, test_file, **extra):
    '''
    '''
    
    label, feat, map, errs = recover_relative_orientation(files, output, **extra)
    rot = orient_utility.map_rotation(feat, map, False)
    format.write_dataset(output, rot, None, label, errs, prefix='rot_raw_')
    _logger.info("Map orthgonal rotation matrices")
    rot = orient_utility.map_rotation(feat, map, True)
    format.write_dataset(output, rot, None, label, errs, prefix='rot_')
    #_logger.info("Convert to Quaternion")
    #rot = orient_utility.ensure_quaternion(rot)
    #format.write_dataset(output, rot, None, label, None)#, prefix='quat_')
    frame, rot = frame_search_mp(files, label, rot, **extra)
    format.write(output, rot, header='psi,theta,phi'.split(','), default_format=format.spiderdoc)
    # write frame
    _logger.info("Completed")
    
def recover_relative_orientation(files, output, neighbors, neighbor_batch, force_embed=False, force_knn=False, symmetric_knn=False, **extra):
    '''
    '''
    
    if force_knn: force_embed=True
    label = build_label(files)
    cache_file = extra.get('cache_file', "")
    feat, evals, index, map, errs = load_from_cache(cache_file, 'feat', 'evals', 'index', 'map', 'errs')
    if feat is None or force_embed:
        dist2, = load_from_cache(cache_file, 'dist2_%d'%neighbors)
        if dist2 is None or force_knn:
            _logger.info("Reading images: %f"%(decimation_level(**extra)))
            if extra['bispec']:
                _logger.info("Using bispec mode: %d"%extra['bispec_mode'])
            mask=create_mask(files, **extra)
            mask;
            data = ndimage_file.read_image_mat(files[0], label, image_transform, mask=None, **extra)
            _logger.info("Dataset: %d,%d"%data.shape)
            if extra['bispec']:
                _logger.info("Using bispectrum representation")
                data -= data.mean(axis=1).reshape((data.shape[0], 1))
                #data /= data.std(axis=1).reshape((data.shape[0], 1))
                #data -= data.min(axis=0)
                #data /= data.max(axis=0)
            _logger.info("Finding nearest neighbors")
            dist2 = manifold.knn(data, neighbors, neighbor_batch)
            save_to_cache(cache_file, **{'dist2_%d'%neighbors: dist2})
        else:
            _logger.info("Using cached nearest neighbors")
        best=(1e20, None)
        program.openmp.set_thread_count(1)
        for err, nn in process_queue.map_reduce_ndarray(orientation_cost_function, extra['thread_count'], range(5,neighbors), dist2, symmetric_knn):#, **extra)
            if err < best[0]: best = (err, nn)
        if best[1] is None: raise ValueError, "Failed to find embedding"
        _logger.info("Best = %d,%f"%(best[1], best[0]))
        program.openmp.set_thread_count(extra['thread_count'])
        feat, evals, index, map, errs=recover_orientation_map(dist2, best[1], not symmetric_knn)
        save_to_cache(cache_file, feat=feat, evals=evals, index=index, map=map, errs=errs)
    else:
        _logger.info("Using cached manifold")
    if index is not None:label = label[index]
    _logger.info("Error: %f - max: %f - min: %f - avg: %f - count: %d"%(numpy.sqrt(errs.sum()), numpy.sqrt(errs.max()), numpy.sqrt(errs.min()), numpy.sqrt(errs.mean()), feat.shape[0]))
    return label, feat, map, errs

def orientation_cost_function(neighbors, dist2, symmetric_knn, **extra):
    '''
    '''
    
    best=(1e20, None)
    for nn in neighbors:
        assert(nn>0)
        try:
            errs = recover_orientation_map(dist2, nn, not symmetric_knn)[-1]
        except:
            err = 1e20
        else:
            err = numpy.mean(errs)
        _logger.info("Neighbor = %d, %f -- %d"%(nn, err, len(errs)))
        if err < best[0]: best = (err, nn)
    return best

def recover_orientation_map(dist2, neighbors, mutual=True, dimension=9):
    '''
    '''
    
    dist2 = manifold.knn_reduce(dist2, neighbors, mutual)
    feat, evals, index = manifold.diffusion_maps_dist(dist2, dimension)
    map, errs = orient_utility.fit_rotation(feat)
    return feat, evals, index, map, errs

def recover_relative_orientation_old(files, output, neighbors, neighbor_batch, **extra):
    '''
    '''
    
    label = build_label(files)
    cache_file = extra.get('cache_file', "")
    feat, evals, index = load_from_cache(cache_file, 'feat', 'evals', 'index')
    if feat is None:
        _logger.info("Reading images: %f"%(decimation_level(**extra)))
        mask=create_mask(files, **extra)
        mask;
        data = ndimage_file.read_image_mat(files[0], label, image_transform, mask=None, **extra)
        if extra['bispec'] and extra['bispec_mode']==0:
            _logger.info("Using bispectrum representation")
            data -= data.min(axis=0)
            data /= data.max(axis=0)
            #assert(data.min(axis=0).shape[0]==data.shape[0])
        _logger.info("Finding nearest neighbors")
        dist2 = manifold.knn(data, neighbors, neighbor_batch)
        _logger.info("Reducing nearest neighbors to 23 mutual neighbors")
        #dist2 = manifold.knn_reduce(dist2, 13, True)
        dist2 = manifold.knn_reduce(dist2, 18, True)
        _logger.info("Embedding manifold: %d"%dist2.data.shape[0])
        feat, evals, index = manifold.diffusion_maps_dist(dist2, 9)
        save_to_cache(cache_file, feat=feat, evals=evals, index=index)
        if index is not None:
            label = label[index]
        format.write_dataset(output, feat, None, label, None, prefix='raw_')
    elif index is not None:
        label = label[index]
    
    map, errs = load_from_cache(cache_file, 'map', 'errs')
    if map is None:
        _logger.info("Fitting manifold")
        map, errs = orient_utility.fit_rotation(feat)
        numpy.square(errs, errs)
        save_to_cache(cache_file, map=map, errs=errs)
    _logger.info("Error: %f - max: %f - min: %f - avg: %f"%(numpy.sqrt(errs.sum()), numpy.sqrt(errs.max()), numpy.sqrt(errs.min()), numpy.sqrt(errs.mean())))
    return label, feat, map, errs

def frame_search_mp(files, label, rot, thread_count=0, cache_file=None, **extra):
    '''
    '''
    
    mask=create_mask(files, **extra)
    frames = numpy.asarray(orient_utility.compute_frames_reject(**extra))
    _logger.info("Reading images")
    images, = load_from_cache(cache_file, 'images')
    if images is None:
        images = ndimage_file.read_image_mat(files[0], label, image_transform, mask=None, compress=False, cache_file=None, dtype=numpy.float32, thread_count=thread_count, **extra)
        save_to_cache(cache_file, images=images)
    _logger.info("Done: %s"%str(images.dtype))
    program.openmp.set_thread_count(1)
    best = (1e20, None)
    
    _logger.info("Ensure Euler")
    rot = numpy.rad2deg(orient_utility.ensure_euler(rot))
    frames = numpy.rad2deg(orient_utility.ensure_euler(frames))
    _logger.error("Frames: %s | %s"%(str(numpy.max(frames, axis=0)), str(numpy.min(frames, axis=0))))
    _logger.error("Rot: %s | %s"%(str(numpy.max(rot, axis=0)), str(numpy.min(rot, axis=0))))
    
    _logger.info("Find frame: %d"%extra['search_grid'])
    for err, frame in process_queue.map_reduce_ndarray(frame_search, thread_count, frames, images, rot, mask, **extra):
        if err < best[0]: best = (err, frame)
    _logger.info("Optimal error: %f"%(best[0]))
    _logger.info("Rotate into frame")
    program.openmp.set_thread_count(thread_count)
    rot=rotate.rotate_euler(best[1], rot)
    _logger.error("Rot: %s | %s"%(str(numpy.max(rot, axis=0)), str(numpy.min(rot, axis=0))))
    return best[1], rot

def frame_search(frames, images, rot, mask, order=5, process_number=0, **extra):
    '''
    '''
    
    midx = numpy.ascontiguousarray(numpy.argwhere(mask.ravel()>0.5).squeeze())
    n = int(numpy.sqrt(float(images.shape[1])))
    curimg = numpy.zeros((1000, n, n), dtype=images.dtype)
    best = (1e20, None)
    for frame in frames:
        err = frame_search_cost_function(frame, images, curimg, rot, midx, order, n)
        if err < best[0]: 
            _logger.info("Best error: %f"%err)
            best = (err, frame)
    
    best = scipy.optimize.fmin(frame_search_cost_function, best[1], full_output=True, args=(images, curimg, rot, midx, order, n))
    return best[1], best[0]

def frame_search_cost_function(frame, images, curimg, rot, midx, order, n):
    '''
    '''
    
    rot=rotate.rotate_euler(frame, rot)
    pix = healpix.ang2pix(order, numpy.deg2rad(rot[:, 1:]))
    #_logger.error("pix=%d"%len(pix))
    #_logger.error("pix=%s"%str(pix))
    #assert(len(pix) > 0)
    idx=numpy.asarray([])
    offset = numpy.min(pix)
    while idx.ndim == 0 or idx.shape[0] <= 100:
        idx = numpy.argwhere(pix==offset).squeeze()
        offset += 1
    idx = idx[:1000]
    assert(len(idx) > 10)
    curimg[:idx.shape[0]] = images[idx].reshape((idx.shape[0], n,n))
    inplane =  -(rot[idx, 0]+rot[idx, 2])
    err = rotate.rotate_error(curimg[:idx.shape[0]], inplane, midx)
    
    if 1 == 0:
        idx = numpy.argwhere(pix==numpy.max(pix)).squeeze()[:1000]
        inplane =  -(rot[idx, 0]+rot[idx, 2])
        curimg[:idx.shape[0]] = images[idx].reshape((idx.shape[0], n,n))
        err += rotate.rotate_error(curimg[:idx.shape[0]], inplane, midx)
    #idx = numpy.argwhere(pix==int(numpy.max(pix)/2)).squeeze()
    return err
    
def angular_distribution(rot, order=6, output="", **extra):
    '''
    '''
    
    import matplotlib
    matplotlib.use("Agg")
    import pylab
    
    res=6
    euler = orient_utility.ensure_euler(rot)
    pix = healpix.ang2pix(res, euler[:, 1:])
    pylab.clf()
    pylab.hist(pix, healpix.res2npix(res))
    pylab.savefig(format_utility.new_filename(output, suffix='_angular_distribution_hist', ext='.png'), dpi=500)

def build_label(files):
    '''
    '''
    
    n = ndimage_file.count_images(files)
    label = numpy.zeros((n, 2))
    end=0
    for filename in files:
        beg=end
        end += ndimage_file.count_images(filename)
        label[beg:end, 0]=spider_utility.spider_id(filename)
        label[beg:end, 1]=numpy.arange(end-beg)
    return label

def create_mask(files, pixel_diameter, **extra):
    '''
    '''
    
    img = ndimage_file.read_image(files[0])
    mask = eman2_utility.model_circle(int(pixel_diameter/2.0), img.shape[0], img.shape[1])
    bin_factor = decimation_level(**extra)
    if bin_factor > 1: mask = eman2_utility.decimate(mask, bin_factor)
    _logger.info("Mask of radius = %d -- width = %d -- bin = %d"%(int(pixel_diameter/2.0), img.shape[0], mask.shape[0]))
    return mask

def image_transform(img, i, mask, var_one=False, bispec=False, compress=True, bispec_mode=1, bispec_rng=[], apix=1.0, **extra):
    '''
    '''
    
    #ndimage_utility.vst(img, img)
    bin_factor = decimation_level(apix=apix, **extra)
    if bin_factor > 1: img = eman2_utility.decimate(img, bin_factor)
    img = eman2_utility.gaussian_low_pass(img, 0.48)
    ndimage_utility.normalize_standard(img, mask, var_one, img)
    if bispec and compress:
        img, freq = ndimage_utility.bispectrum(img, img.shape[0]-1, 'gaussian')
        if len(bispec_rng) > 0:
            idx = numpy.argwhere(numpy.logical_and(freq >= apix/bispec_rng[1], freq <= apix/bispec_rng[0]))
            img = img[idx].copy()
        if bispec_mode == 0:
            img = numpy.log10(numpy.abs(img.real)+1)
        elif bispec_mode == 1:
            img = numpy.angle(img.imag)
            assert(numpy.alltrue(numpy.isfinite(img)))
        elif bispec_mode == 2:
            img = numpy.arctan(img.imag/img.real)
        elif bispec_mode == 3:
            img = numpy.vstack((img.real, img.imag)).T
        else:
            img = numpy.vstack((numpy.log10(numpy.abs(img.real)+1), numpy.angle(img.imag))).T
            
        #bispectrum(signal, maxlag=0.0081, window='gaussian', scale='unbiased')
        freq;
        #img = numpy.log10(numpy.abs(img.real)+1)
    elif mask is not None and compress:
        img = ndimage_utility.compress_image(img, mask)
    return img

def decimation_level(resolution, apix, **extra):
    '''
    '''
    
    return 2
    #dec = resolution / (apix*2)
    #return min(max(dec, 1), 8)
    
def load_from_cache(cache_file, *args):
    '''
    '''
    
    if cache_file is None or cache_file == "":
        _logger.info("No cache file specified")
        return [None for a in args]
    cache_file=os.path.splitext(cache_file)[0]+".mat"
    if  not os.path.exists(cache_file):
        _logger.info("No cache file exists")
        return [None for a in args]
    ret = []
    mat = scipy.io.loadmat(cache_file)
    _logger.info("Cache atlas loaded")
    for filename in args:
        if filename in mat:
            _logger.info("Found cache entry for %s"%filename)
            dtype = mat[filename][0]
            if dtype[0] == '[': dtype = dtype[2:len(dtype)-2]
            if filename+'_sparse' in mat:
                itype=mat[filename+"_sparse"][0]
                if itype[0] == '[': itype = itype[2:len(itype)-2]
                if itype[0] == 'u': itype = itype[2:len(itype)-1]
                _logger.error("itype=%s"%str(itype))
                data = numpy.fromfile(format_utility.new_filename(cache_file, prefix='data_', suffix=filename, ext=".bin"), dtype=numpy.dtype(dtype))
                row = numpy.fromfile(format_utility.new_filename(cache_file, prefix='row_', suffix=filename, ext=".bin"), dtype=numpy.dtype(itype))
                col = numpy.fromfile(format_utility.new_filename(cache_file, prefix='col_', suffix=filename, ext=".bin"), dtype=numpy.dtype(itype))
                val= scipy.sparse.coo_matrix( (data,(row, col)), shape=tuple(mat[filename+"_shape"]) )
            else:
                val = numpy.fromfile(format_utility.new_filename(cache_file, suffix=filename, ext=".bin"), dtype=numpy.dtype(dtype)).reshape(tuple(mat[filename+"_shape"]))
            ret.append(val)
        else:
            _logger.info("No cache entry found for %s"%filename)
            ret.append(None)
    return ret

def save_to_cache(cache_file, **extra):
    '''
    '''
    
    if cache_file == "" or cache_file is None: 
        _logger.info("No file to cache computations")
        return
    cache_file=os.path.splitext(cache_file)[0]+".mat"
    mat = scipy.io.loadmat(cache_file) if os.path.exists(cache_file) else {}
    for key, val in extra.iteritems():
        if val is None: continue
        _logger.info("Writing %s to cache"%key)
        mat[key+"_shape"]=numpy.asarray(val.shape, dtype=numpy.int)
        mat[key] = val.dtype.name
        if scipy.sparse.isspmatrix(val):
            mat[key+"_sparse"]=val.row.dtype.name
            val.data.tofile(format_utility.new_filename(cache_file, prefix='data_', suffix=key, ext=".bin"))
            val.row.tofile(format_utility.new_filename(cache_file, prefix='row_', suffix=key, ext=".bin"))
            val.col.tofile(format_utility.new_filename(cache_file, prefix='col_', suffix=key, ext=".bin"))
        else:
            val.tofile(format_utility.new_filename(cache_file, suffix=key, ext=".bin"))
    scipy.io.savemat(cache_file, mat, oned_as='column', format='5')

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "AutoClass", "Options to control automated classification",  id=__name__)
    group.add_option("", neighbors=30,               help="Maximum number of neighbors to use")
    group.add_option("", resolution=60,              help="Set resolution of orientation recovery")
    group.add_option("", neighbor_batch=50000,        help="Maximum number of neighbors to process in memory at one time: neighbor_batch^2 values", dependent=False)
    group.add_option("", cache_file="",              help="Cache preprocessed data in matlab data files")
    group.add_option("", test_file="",   help="Test nn")
    group.add_option("", search_grid=40,            help="Number of samples for each Euler angle")
    group.add_option("", bispec=False,              help="Use bispectrum of the image")
    group.add_option("", force_embed=False,              help="Force the embedding parameter search")
    group.add_option("", force_knn=False,              help="Force nearest neighbor finding")
    group.add_option("", symmetric_knn=False,        help="Use symmetric (non-mutual) NN graph")
    group.add_option("", bispec_mode=0,             help="Type of bispectrum information")
    group.add_option("", bispec_rng=[],             help="Resolution range for bispectra (low, high)")
    
    pgroup.add_option_group(group)
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with with no digits at the end (e.g. this is bad -> sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        
def check_options(options, main_option=False):
    #Check if the option values are valid
#    from ..core.app.settings import OptionValueError
    
    pass

def main():
    #Main entry point for this script
    program.run_hybrid_program(__name__,
        description = '''Orient projections in 3D space
                        
                        http://
                        
                        Example:
                         
                        $ ara-orient input-stack.spi -p params.spi -o select.dat 
                        
                        Uncomment following to run hereL:
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        use_version = True,
        supports_OMP=True,
    )

def dependents(): return [spider_params]
if __name__ == "__main__": main()


