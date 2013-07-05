'''

http://deeplearning.stanford.edu/wiki/index.php/Implementing_PCA/Whitening

.. Created on Sep 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import matplotlib
matplotlib.use("Agg")

from ..core.app.program import run_hybrid_program
from ..core.image import ndimage_file, eman2_utility, ndimage_utility, reproject, rotate, ndimage_interpolate, manifold #, analysis
from ..core.orient import healpix, orient_utility
from ..core.metadata import spider_utility, format, format_utility, spider_params
#from ..core.parallel import mpi_utility
from ..core.parallel import openmp
import logging, numpy, os, scipy, scipy.cluster.vq, scipy.spatial.distance
import scipy.io, scipy.sparse

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, **extra):
    '''
    '''
    
    _logger.info("Resolution: %f"%extra['resolution'])
    _logger.info("Decimation: %f"%decimation_level(**extra))
    _logger.info("Pixel radius: %d"%(int(extra['pixel_diameter']/2.0)))
    _logger.info("Pixel size: %f"%(extra['apix']))
    _logger.info("Neighbors: %d"%(extra['neighbors']))
    
    if 1 == 1:
        label, ref, align, data = create_sparse_dataset(files, **extra)
    else:
        label, ref, align, data = create_dataset(files, **extra)
    
    _logger.info("Number of samples: %d"%len(label))
    _logger.info("Number of references: %d"%data.shape[1])
    #ref[:] = healpix.ang2pix(healpix_order(**extra), numpy.deg2rad(align[:, 1:3]))
    ref[:] = healpix.ang2pix(1, numpy.deg2rad(align[:, 1:3]))
    feat, index = kernel_pca(data, **extra)
    if index is not None:
        _logger.info("Using subset!")
        label = label[index]
        ref = ref[index]
    format.write_dataset(output, feat, 1, label, ref)
    _logger.info("Completed")
    
def kernel_pca(data, max_eig, neig, max_neighbors=0, sigma=0.0, **extra):
    '''
    '''
    
    # sparsify
    # sparse eigen solver
    # 
    #
    if scipy.sparse.isspmatrix(data):
        if 1 == 1:
            index1 = None
            if max_neighbors > 0:
                data, index1 = manifold.largest_connected(data)
                data = data.tocoo()
                manifold.knn_resort(data)
                #n = data.data.shape[0]
                #data = manifold.knn_mutal(data)
                data = manifold.knn_reduce(data, max_neighbors, True)
                #_logger.error("reduce-safe: %d -> %d"%(n, data.data.shape[0]))
                #assert(data.data.shape[0]>0)
            feat, evals, index = manifold.diffusion_maps_dist(data, max(max_eig, neig), sigma)
            _logger.info("Eigen-values: %s"%",".join([str(e) for e in evals]))
            if index1 is not None:
                if index is not None:
                    index = index1[index]
                else: index = index1
        else:
            pass
    else:
        assert(False)
        manifold.self_tuning_gaussian_kernel_dense(data, True, True)
        # double normalize lik diffusion maps
        U, d, V = scipy.linalg.svd(data, False)
        feat = d*numpy.dot(V, data.T).T
        feat = feat[:, :max(max_eig, neig)]
        index=None
    return feat, index
    
def create_dataset(files, cache_file="", **extra):
    '''
    '''
    
    label, align, ref = read_alignment(files, **extra)
    _logger.info("Read alignment file with %d projections"%len(label))
    if cache_file != "" and (os.path.exists(cache_file) or os.path.exists(cache_file+'.mat')):
        _logger.info("Loading distance matrix from cache")
        mat = scipy.io.loadmat(cache_file)
        data = mat['data']
    else:
        _logger.info("Generating references")
        ref_proj, angles = generate_references(files[0], label, align, **extra)
        _logger.info("Created %d reference projections"%len(ref_proj))
        if cache_file != "":
            _logger.info("Caching references")
            format.write(format_utility.add_prefix(cache_file, 'angles_'), angles, header='psi,theta,phi'.split(','))
            ndimage_file.write_stack(format_utility.add_prefix(cache_file, 'ref_stack_'), ref_proj)
        data = numpy.zeros((len(label), len(ref_proj)), dtype=ref_proj.dtype)
        _logger.info("Calculating distances")
        
        for i, img in enumerate(ndimage_file.iter_images(files[0], label)):
            img = image_transform(img, i, None, align=align, **extra)
            rot = rotate.optimal_inplane(angles, align[i, :3])
            rotate.rotate_distance_array(img, ref_proj, rot, data[i])
        if cache_file != "":
            scipy.io.savemat(cache_file, dict(data=data, label=label), oned_as='column', format='5')
    return label, ref, align, data

def create_sparse_dataset(files, cache_file="", neighbors=2000, batch=10000, eps=1.5, **extra):
    '''
    '''
    
    label, align, refp = read_alignment(files, **extra)
    _logger.info("Read alignment file with %d projections"%len(label))
        
    _logger.info("Generating quaternions")
    ang = align[:, :3].copy()
    #ang[:, 0]=0
    quat=orient_utility.spider_to_quaternion(ang)
    _logger.info("Building nearest neighbor graph: %d"%neighbors)
    qneigh = manifold.knn_geodesic_cache(quat, neighbors, batch, cache_file=cache_file)
    eps = numpy.deg2rad(eps)
    gmax, gmin = manifold.eps_range(qneigh, neighbors)
    _logger.info("Angular distance range for %d neighbors: %f - %f"%(neighbors, numpy.rad2deg(gmin), numpy.rad2deg(gmax)))
    if eps > gmax: 
        _logger.warn("EPS value %f too large must be in range %f - %f"%(numpy.rad2deg(eps), numpy.rad2deg(gmin), numpy.rad2deg(gmax)))
        #raise ValueError, "EPS value %f too large must be in range %f - %f"%(numpy.rad2deg(eps), numpy.rad2deg(gmin), numpy.rad2deg(gmax))
    if eps < gmin: 
        _logger.warn("EPS value %f too small must be in range %f - %f"%(numpy.rad2deg(eps), numpy.rad2deg(gmin), numpy.rad2deg(gmax)))
        #raise ValueError, "EPS value %f too small must be in range %f - %f"%(numpy.rad2deg(eps), numpy.rad2deg(gmin), numpy.rad2deg(gmax))
    epsdata = qneigh.data.copy()
    
    cache_rdat = format_utility.new_filename(cache_file, suffix="_rcoo", ext=".bin") if cache_file != "" else ""
    if cache_rdat != "" and format_utility.os.path.exists(cache_rdat):
        _logger.info("Reading cached revised distances")
        qneigh.data[:] = numpy.fromfile(cache_rdat, dtype=qneigh.data.dtype)
        _logger.info("Image data: %f-%f"%(numpy.min(qneigh.data[:neighbors+1]), numpy.max(qneigh.data[:neighbors+1])))
        _logger.info("Image data2: %f-%f"%(numpy.min(qneigh.data[neighbors+1:][:neighbors+1]), numpy.max(qneigh.data[neighbors+1:][:neighbors+1])))
        _logger.info("Angle data: %f-%f"%(numpy.rad2deg(numpy.min(epsdata[:neighbors+1])), numpy.rad2deg(numpy.max(epsdata[:neighbors+1]))))
    else:
        _logger.info("Creating mask")
        mask = create_mask(files, **extra)
        _logger.info("Reading images from disk")
        openmp.set_thread_count(1)
        samp = ndimage_file.read_image_mat(files[0], label, image_transform, False, cache_file, align=align, mask=mask, dtype=numpy.float32, **extra)
        openmp.set_thread_count(extra['thread_count'])
        mask = numpy.ascontiguousarray(numpy.argwhere(mask.ravel() > 0).squeeze())
        if 1 == 1:
            samp = samp.astype(numpy.float32)
            n = int(numpy.sqrt(samp.shape[1]))
            samp1 = samp.reshape((len(samp), n, n))
            _logger.info("Calculating change in rotation using %d threads"%extra['thread_count'])
            openmp.set_thread_count(1)
            #rot = orient_utility.optimal_inplace_rotation_mp(quat, qneigh.row, qneigh.col, extra['thread_count'])
            rot = orient_utility.optimal_inplace_rotation_mp(align[:, :3], qneigh.row, qneigh.col, extra['thread_count'])
            openmp.set_thread_count(extra['thread_count'])
            #rot = orient_utility.optimal_inplace_rotation2(align[:, :3], qneigh.row, qneigh.col)
            _logger.info("Recalculating distances: %s"%str(samp.dtype))
            rotate.calc_rotated_distance_mask(samp1, qneigh.row, qneigh.col, rot, qneigh.data, mask)
        else:
            _logger.info("Calculating restricted")
            manifold.knn_restricted(qneigh, samp, mask)
            
        _logger.info("Reducing with epsilon nearest neighbor: %f"%eps)
        qneigh.data.tofile(cache_rdat)
    _logger.info("Reduce angular neighborhood: %f (%f)"%(eps, numpy.rad2deg(eps)))
    qneigh = manifold.knn_reduce_eps(qneigh, eps, epsdata) #.tocsr()
    if 1 == 0:
        sigma = scipy.linspace(-8, 8, 100)
        kernel_cum = numpy.zeros((2, sigma.shape[0]), dtype=qneigh.data.dtype)
        kernel_cum[0, :]=10**sigma
        manifold._manifold.gaussian_kernel_range(qneigh.data, kernel_cum)
        for i in xrange(len(sigma)):
            _logger.info("%f,%f"%(sigma[i], kernel_cum[1, i]))
    _logger.info("Image data2: %f-%f"%(numpy.min(qneigh.data[:neighbors+1]), numpy.max(qneigh.data[:neighbors+1])))
    _logger.info("Angle data2: %f-%f"%(numpy.rad2deg(numpy.min(epsdata[:neighbors+1])), numpy.rad2deg(numpy.max(epsdata[:neighbors+1]))))
    return label, refp, align, qneigh

def generate_references(filename, label, align, reference="", **extra):
    '''
    '''
    
    if reference == "":
        return generate_reference_averages(filename, label, align, **extra)
    else:
        return generate_reference_projections(reference, **extra)

def generate_reference_averages(filename, label, align, **extra):
    '''
    '''
    
    resolution = healpix_order(**extra)
    _logger.info("Resolution: %f"%extra['resolution'])
    _logger.info("Decimation: %f"%decimation_level(**extra))
    _logger.info("Healpix Order: %d -> %f deg. -> %d"%(resolution, healpix.nside2pixarea(resolution, True), healpix.res2npix(resolution)))
    angs = healpix.angles(resolution)
    resolution = pow(2, resolution)
    avgs = None
    counts = numpy.zeros(len(angs))
    for i, img in enumerate(ndimage_file.iter_images(filename, label)):
        img = image_transform(img, i, None, align=align, **extra)
        if avgs is None:
            avgs = numpy.zeros((len(angs), img.shape[0], img.shape[1]), dtype=img.dtype)
        ipix = healpix._healpix.ang2pix_ring(resolution, numpy.deg2rad(align[i, 1]), numpy.deg2rad(align[i, 2]))
        rot = rotate.optimal_inplane(angs[ipix], align[i, :3])
        #rang = rotate.rotate_euler(ang[ipix], align[i, :3])
        #rot = -(rang[0]+rang[2])
        avgs[ipix] += rotate.rotate_image(img, rot)
        counts[ipix]+=1
    for i in xrange(len(avgs)):
        if counts[i] > 0:
            avgs[i]/=counts[i]
            avgs[i]-=avgs[i].mean()
    #avg /= counts
    #avg -= avg.mean(axis=1)
    return avgs, angs
    
def generate_reference_projections(reference, pixel_diameter, **extra):
    '''
    '''
    
    reference = ndimage_file.read_image(reference)
    bin_factor = decimation_level(**extra)
    _logger.info("Decimate data by %f for resolution %f"%(bin_factor, extra['resolution']))
    if bin_factor > 1: reference = ndimage_interpolate.interpolate_bilinear(reference, bin_factor) #reference = eman2_utility.decimate(reference, bin_factor)
    # decimate to window size
    # normalize projections
    angles = healpix.angles(healpix_order(pixel_diameter, **extra))
    _logger.info("Created %d angles from healpix order %d"%(len(angles), healpix_order(pixel_diameter, **extra)))
    return reproject.reproject_3q(reference, pixel_diameter/2, angles, **extra), angles

def read_alignment(files, alignment="", is_dala=False, **extra):
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
        raise ValueError, "Multi file alignment read not supported"
    if is_dala:
        align[:, 0]=-align[:, 5]
    ref = align[:, refidx].astype(numpy.int)
    return label, align, ref
    
def create_mask(files, pixel_diameter, **extra):
    '''
    '''
    
    img = ndimage_file.read_image(files[0])
    mask = eman2_utility.model_circle(int(pixel_diameter/2.0), img.shape[0], img.shape[1])
    #bg = eman2_utility.model_circle(int(pixel_diameter/2.0), img.shape[0], img.shape[1])*-1+1
    #bg[bg > 0] = numpy.mean(img[mask>0])
    bin_factor = decimation_level(**extra)
    #_logger.info("Decimation factor %f for resolution %f and pixel size %f"%(bin_factor,  resolution, apix))
    if bin_factor > 1: 
        mask = eman2_utility.decimate(mask, bin_factor)
        #bg = eman2_utility.decimate(bg, bin_factor)
    return mask #, bg

def image_transform(img, i, mask, var_one=True, align=None, bispec=False, **extra):
    '''
    '''
    
    #if align[i, 1] > 179.999: img = eman2_utility.mirror(img)
    #if align[i, 0] != 0: img = eman2_utility.rot_shift2D(img, align[i, 0], 0, 0, 0)
    ndimage_utility.vst(img, img)
    bin_factor = decimation_level(**extra)
    if bin_factor > 1: img = eman2_utility.decimate(img, bin_factor)
    ndimage_utility.normalize_standard(img, mask, var_one, img)
    '''
    if bispec:
        img, freq = ndimage_utility.bispectrum(img, img.shape[0]-1, 'gaussian')
        #bispectrum(signal, maxlag=0.0081, window='gaussian', scale='unbiased')
        freq;
        img = numpy.log10(numpy.abs(img.real)+1)
    '''
    #if mask is not None:
    #    img = ndimage_utility.compress_image(img, mask)
    if mask is not None: img *= mask
    return img

def image_transform_sim(img, i, mask, var_one=True, align=None, bispec=False, apix=2.81, resolution=12.0, **extra):
    '''
    '''
    
    if resolution > 0 and apix > 0: img = eman2_utility.gaussian_low_pass(img, apix/resolution, 1)
    img = eman2_utility.normalize_mask(img, mask)
    return img

def decimation_level(resolution, apix, window, **extra):
    '''
    '''
    
    #if 1==1: return 1.0
    dec = resolution / (apix*3)
    d = float(window)/dec + 10
    d = window/float(d)
    return min(max(d, 1), 8)
    
    #return max(1, min(8, resolution / (apix*2))) if resolution > (2*apix) else 1

def healpix_order(pixel_diameter, resolution, apix, **extra):
    '''
    '''
    
    return healpix.theta2nside( ( numpy.arctan( resolution / (pixel_diameter*apix) ) ) * 2 )
    

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "AutoClass", "Options to control automated classification",  id=__name__)
    group.add_option("", neig=1,                    help="Number of eigen vectors to use", dependent=False)
    group.add_option("", max_eig=30,                help="Maximum number of eigen vectors saved")
    #group.add_option("", nstd=2.5,                    help="Number of deviations from the median", dependent=False)
    #group.add_option("", bispec=False,                  help="Enable bispectrum feature space")
    group.add_option("", resolution=20,              help="Enable bispectrum feature space")
    group.add_option("", cache_file="",              help="Cache preprocessed data in matlab data files")
    group.add_option("", eps=1.5,              help="Angular step size")
    group.add_option("", neighbors=2000,              help="Number of neighbors")
    group.add_option("", batch=10000,              help="Maximum partial distance matrix")
    group.add_option("", is_dala=False,              help="Set true if in-plane rotation has been applied")
    group.add_option("", max_neighbors=0,              help="Maximum number of neighbors")
    group.add_option("", sigma=0.0,              help="Kernel parameter (0.0 means use automatica local-scaling)")
    #group.add_option("", sigma_range=)
    
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with with no digits at the end (e.g. this is bad -> sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-a", alignment="",   help="Input file containing alignment parameters", required_file=True, gui=dict(filetype="open"))
        #pgroup.add_option("-r", reference="",   help="Input file containing a reference volume", required_file=True, gui=dict(filetype="open"))

def check_options(options, main_option=False):
    #Check if the option values are valid
#    from ..core.app.settings import OptionValueError
    
    pass

def main():
    #Main entry point for this script
    run_hybrid_program(__name__,
        description = '''Classify particles with a reference
                        
                        http://
                        
                        Example:
                         
                        $ ara-autoclass input-stack.spi -p params.spi -r reference -a align.spi -o select.dat 
                        
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

