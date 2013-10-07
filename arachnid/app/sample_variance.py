''' Estimate the variance in a 3D density map using uniform view subsampling



.. Created on Feb 28, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.metadata import spider_params, format_alignment, spider_utility, format_utility
from ..core.parallel import mpi_utility, process_tasks
from ..core.image import ndimage_file, reconstruct, eman2_utility
from ..core.util import numpy_ext
from ..core.orient import healpix
import logging, numpy, itertools, os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output="", **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
    
    files : list
            List of input filenames
    output : str
             Output filename for refinement
    extra : dict
            Unused keyword arguments
    '''
    
    # cache by decimating to resolution?
    # dala or shift?
    # phase flip?
    # apply shift, rotation to data
    
    
    filename = files[0]
    id_len = extra.get('id_len', 0)
    if mpi_utility.get_size(**extra) > 1:
        volume_count = extra['volume_count']
        extra['volume_count'] /= mpi_utility.get_size(**extra)
        if mpi_utility.is_root(**extra):
            _logger.info("Sampling %d volumes over %d nodes with %d volumes per node"%(volume_count, mpi_utility.get_size(**extra), extra['volume_count']))
    else:
        _logger.info("Sampling %d volumes"%(extra['volume_count']))
    
    for filename in files:
        if len(files) > 0:
            output = spider_utility.spider_filename(output, filename, id_len)
        estimate_sample_variance(filename, output, **extra)
    
    if mpi_utility.is_root(**extra): _logger.info("Completed")
    
def estimate_sample_variance(filename, output, sample_size, thread_count, volume_count=1000, fast=False, twopass=True, **extra):
    ''' 
    '''
    
    image_files, align = format_alignment.read_alignment(filename, use_3d=False, **extra)
    image_size = ndimage_file.read_image(image_files).shape[0] if isinstance(image_files, str) else ndimage_file.read_image(image_files[0][0]).shape[0]
    if sample_size == 0: sample_size = len(align)
    elif sample_size < 1.0: sample_size = int(sample_size*len(align))
    if thread_count < 1: thread_count=1
    size = mpi_utility.get_size(**extra)
    if size < 1: size=1
    
    if mpi_utility.is_root(**extra):
        replacement = "with" if sample_size == len(align) else "without"
        _logger.info("Sampling %d projections per volume %s replacement"%(sample_size, replacement))
        _logger.info("Using HEALPix order %d - %d degree sampling"%(extra['view_resolution'], healpix.nside2pixarea(extra['view_resolution'], True)))
    
    gavg = None
    tv_cnt = volume_count/thread_count
    if not fast or twopass:
        gavg_output = format_utility.add_prefix(output, 'gavg_')
        if os.path.exists(gavg_output) and 1 == 0:
            gavg = ndimage_file.read_image(gavg_output)
        else:
            for a in process_tasks.process_queue.start_reduce(sample_average, thread_count, image_files, image_size, align, sample_size, tv_cnt, **extra):
                if gavg is None: gavg=a
                else: gavg += a
            mpi_utility.block_reduce(gavg, **extra)
            gavg /= (volume_count*size)
            ndimage_file.write_image(gavg_output, gavg)
    
    avg = None
    avg2 = None
    
    if twopass:
        for a, a2 in process_tasks.process_queue.start_reduce(sample_variance_two_pass, thread_count, image_files, image_size, align, sample_size, tv_cnt, gavg=gavg, **extra):
            if avg is None:
                avg = a
                avg2 = a2
            else:
                avg += a
                avg2 += a2
        
        for a, a2 in mpi_utility.iterate_reduce((avg, avg2), **extra):
            avg += a
            avg2 += a2
        if mpi_utility.is_root(**extra):
            var=avg.copy()
            #var = (avg - avg2**2/volume_count)
            #var = (avg2**2/volume_count-avg)
    else:
        assert(False)
        for a, a2 in process_tasks.process_queue.start_reduce(sample_variance, thread_count, image_files, image_size, align, sample_size, tv_cnt, gavg=gavg, **extra):
            if avg is None:
                avg = a
                avg2 = a2
                ndimage_file.write_image(format_utility.add_prefix(output, 'tmp1_'), avg)
                ndimage_file.write_image(format_utility.add_prefix(output, 'tmp2_'), avg2)
            else:
                avg = (tv_cnt*avg+tv_cnt*a)/(2*tv_cnt)
                avg2 = ((tv_cnt*avg2)+(tv_cnt*a2))/(2*tv_cnt) + ((tv_cnt*tv_cnt)*((avg2-a2)/(2*tv_cnt))**2)
        
        for a, a2 in mpi_utility.iterate_reduce((avg, avg2), **extra):
            avg = (volume_count*avg+volume_count*a)/(2*volume_count)
            avg2 = ((volume_count*avg2)+(volume_count*a2))/(2*volume_count) + ((volume_count*volume_count)*((avg2-a2)/(2*volume_count))**2)
        var = avg2
    #mean_ab = ((mean_a * n_a) + (mean_b * n_b)) / n_ab
    #var_ab = (((n_a * var_a) + (n_b * var_b)) / n_ab) + ((n_a * n_b) * ((mean_b - mean_a) / n_ab)**2)
    
    if mpi_utility.is_root(**extra):
        avg_output = format_utility.add_prefix(output, 'avg_')
        avg2_output = format_utility.add_prefix(output, 'avg2_')
        cnt_output = os.path.splitext(format_utility.add_prefix(output, 'cnt_'))[0]
        
        #volume_count*size
        
        #TODO: read volumes and count, recombine
        
        if avg is not None:
            ndimage_file.write_image(avg_output, avg)
        ndimage_file.write_image(avg2_output, avg2)
        numpy.save(cnt_output, numpy.asarray([volume_count*size], dtype=numpy.int))
        
        
        var /= (volume_count*size-1)
        ndimage_file.write_image(output, var)

def sample_variance_two_pass(image_files, image_size, align, sample_size, volume_count, gavg, reconstruct_type='bp3f', thread_count=1, cache_file="", **extra):
    '''
    '''
    
    index = numpy.arange(len(align), dtype=numpy.int)
    weights = sample_weights(align, **extra)
    if sample_size == 0: sample_size = len(index)
    elif sample_size < 1.0: sample_size = int(sample_size*len(index))
    if cache_file != "":
        cache_file = mpi_utility.safe_tempfile(cache_file, False)
    avg = None
    avg2 = None
    gavg -= gavg.mean()
    for i in xrange(volume_count):
        if cache_file == "":
            assert(False)
            sindex = sample(index, weights, sample_size)
            vol = reconstruct_vol(image_files, align, sindex, image_size, reconstruct_type, thread_count)
        else:
            if cache_file != "": vol=ndimage_file.read_image(cache_file, i)
        #vol = gavg - vol
        vol = filter_volume(vol, **extra)
        vol -= vol.mean()
        vol -= gavg
        if avg is None:
            avg2 = vol.copy()
            avg = numpy.square(vol)
        else:
            avg2 += vol
            avg += numpy.square(vol)
    return avg, avg2

def sample_variance(image_files, image_size, align, sample_size, volume_count, gavg=None, reconstruct_type='bp3f', thread_count=1, **extra):
    '''
    '''
    
    index = numpy.arange(len(align), dtype=numpy.int)
    weights = sample_weights(align, **extra)
    if sample_size == 0: sample_size = len(index)
    elif sample_size < 1.0: sample_size = int(sample_size*len(index))
    avg = None
    avg2 = None
    for i in xrange(volume_count):
        sindex = sample(index, weights, sample_size)
        vol = reconstruct_vol(image_files, align, sindex, image_size, reconstruct_type, thread_count)
        if gavg is not None: vol -= gavg
        if avg is None:
            avg = vol.copy()
            avg2 = vol.copy()
        else:
            delta = vol - avg
            avg += (delta/(i+1))
            avg2 += delta*(vol-avg)
    return avg, avg2
    
def sample_average(image_files, image_size, align, sample_size, volume_count, reconstruct_type='bp3f', thread_count=1, cache_file="", **extra):
    '''
    '''
    
    index = numpy.arange(len(align), dtype=numpy.int)
    weights = sample_weights(align, **extra)
    if sample_size == 0: sample_size = len(index)
    elif sample_size < 1.0: sample_size = int(sample_size*len(index))
    
    if cache_file != "":
        cache_file = mpi_utility.safe_tempfile(cache_file, False, **extra)
    
    avg = None
    for i in xrange(volume_count):
        sindex = sample(index, weights, sample_size)
        if cache_file != "" and os.path.exists(cache_file):
            vol=ndimage_file.read_image(cache_file, i)
        else:
            assert(False)
            vol = reconstruct_vol(image_files, align, sindex, image_size, reconstruct_type, thread_count)
            if cache_file != "": ndimage_file.write_image(cache_file, vol, i)
        vol = filter_volume(vol, **extra)
        if avg is None: avg = vol
        else: avg += vol
    return avg

def filter_volume(vol, resolution=0, apix=0, **extra):
    '''
    '''
    
    if resolution > 0 and apix > 0:
        vol = eman2_utility.gaussian_low_pass(vol, apix/resolution)
    return vol

def sample_weights(align, view_resolution, view_eps, **extra):
    '''
    '''
    
    view = healpix.ang2pix(view_resolution, numpy.deg2rad(align[:, 1:3]))
    _logger.info("min: %d -- max: %d -- count: %d"%(view.min(), view.max(), healpix.res2npix(view_resolution)))
    vhist = numpy.histogram(view, healpix.res2npix(view_resolution))[0]
    min_view = numpy.mean(vhist)*view_eps
    weights = numpy.zeros(len(align))
    for i in xrange(len(align)):
        weights[i] = 1.0/min(vhist[view[i]], min_view)
    weights /= weights.sum()
    return weights

def sample(index, weights, sample_size):
    '''
    '''
    
    return numpy_ext.choice(index, size=sample_size, replace=(sample_size==len(index)), p=weights)
    
def reconstruct_vol(files, align, index, image_size, reconstruct_type='bp3f', thread_count=1):
    '''
    '''
    
    if isinstance(files, str):
        iter_images = ndimage_file.iter_images(files, index)
    else:
        iter_files = itertools.imap(lambda i: files[i], index)
        iter_images = itertools.imap(ndimage_file.read_image, iter_files)
    
    if reconstruct_type=='bp3f':
        vol = reconstruct.reconstruct_bp3f_mp(iter_images, image_size, align[index], thread_count=1)
    else:
        vol = reconstruct.reconstruct_bp3n_mp(iter_images, image_size, align[index], thread_count=1)
    return vol
    #ndimage_file.write_image(output, vol)

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import OptionGroup
    
    if main_option:        
        bgroup = OptionGroup(parser, "Primary", "Primary options to set for input and output", group_order=0,  id=__name__)
        bgroup.add_option("-i", input_files=[],         help="List of input alignment files", required_file=True, gui=dict(filetype="file-list"))
        bgroup.add_option("-o", output="",              help="Base filename for output volume and half volumes, which will be named raw_$output, raw1_$output, raw2_$output", gui=dict(filetype="save"), required_file=True)
        bgroup.add_option("-m", image_file="",          help="Filename for the image stack template", gui=dict(filetype="open"), required_file=True)
        pgroup.add_option_group(bgroup)
        
    group = OptionGroup(parser, "Additional", "Options to customize your refinement", group_order=0,  id=__name__)
    group.add_option("",   view_resolution=3,   help="Coarse-grain view resolution using HealPix parameters")
    group.add_option("",   resolution=0,        help="Resolution to filter the volumes")
    group.add_option("",   volume_count=10000,  help="Number of reconstructions to perform")
    group.add_option("",   sample_size=0.5,     help="Number of images to sample per reconstruction: <1 is treated percentage, 0 is treated as with replacement")
    group.add_option("",   view_eps=0.1,        help="Regularization parameter for the view weights")
    group.add_option("",   fast=False,          help="Do not precompute the mean volume")
    group.add_option("",   cache_file="",       help="Directory to store volume cache")
    #group.add_option("-t", thread_count=1,    help="Number of parallel reconstructions")
    #group.add_option("-w", worker_count=0,      help="Set number of  workers to process volumes in parallel",  gui=dict(maximum=sys.maxint, minimum=0), dependent=False)
    #group.add_option("",   keep_volumes=False,  help="Keep all volumes in a large stack")
    pgroup.add_option_group(group)
    
    if main_option:
        parser.change_default(log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if options.sample_size < 0: raise OptionValueError, "--sample-size must be 0 or greater"
    if options.volume_count <= 0: raise OptionValueError, "--volume-count must be greater than 0"
    
def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Estimate the variance in a 3D density map using uniform view subsampling
                        
                        $ %prog align.spi -p params.ter -m stack_template.ter -o var_map.spi
                        
                        http://
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                        
                        a cluster:
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nodes=`python -c "fin=open('machinefile', 'r');lines=fin.readlines();print len([val for val in lines if val[0].strip() != '' and val[0].strip()[0] != '#'])"`
                        nohup mpiexec -stdin none -n $nodes -machinefile machinefile %prog -c $PWD/$0 --use-MPI < /dev/null > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_OMP=True,
        supports_MPI=True,
        use_version = True,
    )
def dependents(): return [spider_params]
if __name__ == "__main__": main()

