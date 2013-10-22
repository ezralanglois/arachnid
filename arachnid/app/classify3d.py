''' Classifies projections based on 3D hetergenity

This script (`ara-classify3d`) classifies projections based on 3D hetergenity.

Tips
====
 
 #. Output filename: Must follow the SPIDER format with a number before the extension, e.g. mic_00001.spi. Output files just require the number of digits: `--output sndc_0000.spi`

Running Script
===============

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    $ ara-classify3d win*.spi -a align.spi -o selection.spi

Critical Options
================

.. program:: ara-selrelion

.. option:: -i <filename1,filename2>, --input-files <filename1,filename2>, filename1 filename
    
    List of filenames for the input micrographs.
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`

.. option:: -o <str>, --output <str>
    
    Output filename for the relion selection file

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`

.. Created on Nov 13, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.metadata import spider_utility, format_utility, format, spider_params
from ..core.image import ndimage_file, manifold, ndimage_utility, ndimage_interpolate, ndimage_processor
from ..core.orient import orient_utility, transforms
from ..core.parallel import process_queue
import logging, os, numpy, scipy.ndimage.interpolation
try: 
    import psutil
    psutil;
except: psutil=None

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, neighbors, dimension, batch, eps, **extra):
    '''Convert a set of micrographs to SPIDER
    
    :Parameters:
    
    filename : list
               List of micrographs names
    output : str
             Output SPIDER filename template
    extra : dict
            Unused key word arguments
    '''
    
    # 1. Read alignment file
    _logger.info("Reading alignment file: %s"%extra['alignment'])
    label, align = read_alignment(files, **extra)
    cache_file = format_utility.add_prefix(output, "cache_")
    
    shared=False
    # 2. calculate neighbor based on alignment
    _logger.info("Estimating nearest neighbors: %d"%neighbors)
    quat, shm_quat = process_queue.create_global_dense_matrix((len(align), 4), shared=shared)
    orient_utility.spider_to_quaternion(align[:, :3], quat)
    qneigh = manifold.knn_geodesic_cache(quat, neighbors, batch, shared=shared, cache_file=cache_file)
    
    eps = numpy.deg2rad(eps)
    gmax, gmin = manifold.eps_range(qneigh, neighbors)
    _logger.info("Angular distance range for %d neighbors: %f - %f"%(neighbors, numpy.rad2deg(gmin), numpy.rad2deg(gmax)))
    if eps > gmax:
        raise ValueError, "EPS value %f too large must be in range %f - %f"%(numpy.rad2deg(eps), numpy.rad2deg(gmin), numpy.rad2deg(gmax))
    if eps < gmin:
        raise ValueError, "EPS value %f too small must be in range %f - %f"%(numpy.rad2deg(eps), numpy.rad2deg(gmin), numpy.rad2deg(gmax))
    _logger.info("Angular2 distance range for %d neighbors: %f - %f - %f"%(neighbors, gmin, eps, gmax))
    
    
    _logger.info("Reading %d images"%len(label))
    mask = create_mask(files, **extra)
    shm_data = ndimage_processor.read_image_mat(files, label, image_transform, shared=shared, mask=mask, cache_file=cache_file, **extra)
    
    _logger.info("Revising distances with %d threads"%extra.get('thread_count', 1))
    # 4. replace distances with rotated distance - rotate align so all projections spiral out
    epsdata = qneigh.data.copy()
    recalculate_distance(qneigh, shm_quat, shm_data, mask, neighbors, **extra)
    qneigh = manifold.knn_reduce_eps(qneigh, eps, epsdata)
    _logger.info("Check: %f"%qneigh.data[1])
    
    _logger.info("Embedding diffusion maps into %d dimensions"%dimension)
    # 5. embed manifold
    #qneigh = manifold.knn_reduce(qneigh, neighbors, True)
    feat, evals, index = manifold.diffusion_maps_dist(qneigh, dimension)
    _logger.info("Eigen values: %s"%",".join([str(v) for v in evals]))
    # 6. Write data
    feat_old = feat
    feat = numpy.zeros((len(label), label.shape[1]+feat.shape[1]))
    feat[:, :label.shape[1]]=label
    if index is not None: feat[index, label.shape[1]:] = feat_old
    else: feat[:, label.shape[1]:] = feat_old
    header = []
    for i in xrange(label.shape[1]): header.append("id_%d"%(i+1))
    for i in xrange(feat_old.shape[1]): header.append("c%d"%(i+1))
    
    _logger.info("Writing manifold to %s - %d == %d"%(output, feat.shape[1], len(header)))
    format.write(output, feat, default_format=format.csv, header=header)
    _logger.info("Completed")
    
def recalculate_distance(shm_qneigh, shm_quat, shm_data, mask, neighbors, thread_count, **extra):
    '''
    '''
    
    for beg, end, data in process_queue.map_reduce_array(recalculate_distance_worker, thread_count, shm_data, shm_qneigh, shm_quat, mask, neighbors):
        shm_qneigh.data[beg:end]=data

def recalculate_distance_worker(beg, end, shm_data, shm_qneigh, shm_quat, mask, neighbors, process_number=0, **extra):
    '''
    '''
    
    qneigh = process_queue.recreate_global_sparse_matrix(shm_qneigh)
    quat = process_queue.recreate_global_dense_matrix(shm_quat)
    data = process_queue.recreate_global_dense_matrix(shm_data)
    euler = numpy.zeros(3)
    nn = neighbors+1
    n = numpy.sqrt(data.shape[1])
    
    start = beg*nn
    mat_data = numpy.zeros((end-beg)*nn)
    offset = (end-beg)/10
    for r in xrange(beg, end):
        if ((r-beg)%offset) == 0:
            if psutil is not None:
                _logger.info("Process %d is %f done with %.2f MB memory usage"%(process_number, (float(r-beg)/(end-beg)*100), psutil.Process(os.getpid()).get_memory_info().rss/131072 ))
            else:
                _logger.info("Process %d is %f\% done"%(process_number, ((r-beg)/(end-beg)*100)))
        b = nn*r
        assert(r==qneigh.col[b])
        euler[:] = transforms.euler_from_quaternion(quat[r], 'rzyz')
        #frame = transforms.quaternion_inverse(quat[qneigh.col[b]])
        frame = quat[qneigh.col[b]]
        #if not numpy.alltrue(numpy.isfinite(euler)): raise ValueError, "Non finite values detected in array"
        rdata = data[r].reshape((n,n))
        if euler[0] != 0: 
            rdata = scipy.ndimage.interpolation.rotate(rdata, -euler[0], mode='wrap')
            diff = (rdata.shape[0]-n)/2
            rdata = rdata[diff:(rdata.shape[0]-diff), diff:(rdata.shape[0]-diff)]
        if process_number == 0 and r < (beg+100):
            avg = rdata.copy()
        
        rdata = ndimage_utility.compress_image(rdata, mask)
        for j in xrange(neighbors):
            c = qneigh.col[b+1+j]
            euler[:] = numpy.rad2deg(transforms.euler_from_quaternion(transforms.quaternion_multiply(quat[c], frame), 'rzyz'))
            #if not numpy.alltrue(numpy.isfinite(euler)): raise ValueError, "Non finite values detected in array"
            cdata = scipy.ndimage.interpolation.rotate(data[c].reshape((n,n)), -(euler[0]+euler[2]), mode='wrap')
            diff = (cdata.shape[0]-n)/2
            cdata = cdata[diff:(cdata.shape[0]-diff), diff:(cdata.shape[0]-diff)]
            if process_number == 0 and r < (beg+100):
                avg+=cdata
            cdata = ndimage_utility.compress_image(cdata, mask)
            mat_data[b+1+j-start] = numpy.sum(numpy.square(cdata-rdata))
            #qneigh.data[b+1+j] = numpy.sum(numpy.square(cdata-rdata)) 212 - 356 - 3800
        if process_number == 0 and r < (beg+500):
            ndimage_file.write_image("avg_test01.spi", avg, int(r-beg))
    return beg*nn, end*nn, mat_data

def create_mask(files, pixel_diameter, resolution, apix, **extra):
    '''
    '''
    
    img = ndimage_file.read_image(files[0])
    mask = ndimage_utility.model_disk(int(pixel_diameter/2.0), img)
    bin_factor = max(1, min(8, resolution / (apix*2))) if resolution > (2*apix) else 1
    _logger.info("Decimation factor %f for resolution %f and pixel size %f"%(bin_factor, resolution, apix))
    if bin_factor > 1: mask = ndimage_interpolate.downsample(mask, bin_factor)
    return mask

def image_transform(img, i, mask, resolution, apix, var_one=True, **extra):
    '''
    '''
    
    ndimage_utility.vst(img, img)
    bin_factor = max(1, min(8, resolution / (apix*2))) if resolution > (2*apix) else 1
    if bin_factor > 1: img = ndimage_interpolate.downsample(img, bin_factor)
    ndimage_utility.normalize_standard(img, mask, var_one, img)
    #img = ndimage_utility.compress_image(img, mask)
    return img
    
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
    return label, align

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Classify 3D", "Options to control classifcation",  id=__name__)
    group.add_option("-a", alignment="", help="Input alignment filename", required_file=True, gui=dict(filetype="open"))
    group.add_option("-n", neighbors=10, help="Number of neighbors", gui=dict(minimum=1))
    group.add_option("-r", resolution=10, help="Resolution of the structure", gui=dict(minimum=1))
    group.add_option("-d", dimension=5,   help="Number of dimensions in manifold space", gui=dict(minimum=1))
    group.add_option("",   batch=10000,   help="Number of examples to hold in memory for partial distance matrix (batch*batch)", gui=dict(minimum=1))
    group.add_option("",   eps=1.0,       help="Largest angular distance allowed for neighbor graph (degrees)")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input stacks or selection file", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the relion selection file", gui=dict(filetype="save"), required_file=True)
        parser.change_default(log_level=3)


def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    if options.neighbors < 1: raise OptionValueError, "Neighbors must be positive integer"


def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Classifies projections based on 3D hetergenity
        
                         http://
                         
                         $ ara-classify3d win*.spi -a align.spi -o selection.spi
                      ''',
        supports_OMP=True,
        supports_MPI = False,
        use_version = False,
    )
def dependents(): return [spider_params]
if __name__ == "__main__": main()


