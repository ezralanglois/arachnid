''' Automated passive-aggressive angular refinement

This |spi| batch file (`spi-refine`) refines the orientational assignment of a set of projections against a
continually improving reference. Unlike traditional refinement algorithms, this script chooses all the 
parameters for the user based on a few rules of thumb.

Tips
====

 #. If the data stack has already been phase flipped, then set `phase_flip` to True
 
 #. Be aware that if the data or the reference does not match the size in the params file, it will be automatically reduced in size
    Use --bin-factor to control the decimation of the params file, and thus the reference volumes, masks and data stacks.

 #. When using MPI, :option:`home-prefix` should point to a directory on the head node, :option:`local-scratch` should point to a local directory 
    on the cluster node, :option:`shared-scratch` should point to a directory accessible to all nodes
    and all other paths should be relative to :option:`home-prefix`.

 #. Any parameter in the configuration file can be changed during refinement. Simply list it amoung the other parameters in `refine-name` and set its value
    in the corresponding position. If you want to change the third parameter in this list, you must specifiy the first two. Any parameter after the third, 
    which is not listed will either take the value from the previous iteration of refinement or from the value in the configuration file.
     
Examples
========

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Run alignment over a phase flipped stack
    
    $ spi-refine image_stack_*.ter -p params.ter -o alignment_0001.ter -r ref_vol_0001.spi --phase-flip

Critical Options
================

.. program:: spi-refine

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames containing stacks of projection images
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`
    
.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for the raw full and half volumes as well as base output name for FSC curve (`res_$output`)

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Path to SPIDER params file

.. option:: --phase-flip <BOOL> 
    
    Set to True if your data stack(s) has already been phase flipped or CTF corrected (Default: False)

.. option:: -r <FILENAME>, --reference <FILENAME>
    
    Filename for reference with the proper pixel size

.. option:: -a <FILENAME>, --alignment <FILENAME> 
    
    Filename for the alignment parameters
    
Useful Options
==============

.. option:: --refine-index <INT>
    
    Iteration to start refinment: -1 = start at last volume, 0 = start at begining, > 0 start after specific iteration (Default: -1)
    
.. option:: --resolution-start <FLOAT>
    
    Starting resolution for the refinement (Default: 30.0)

.. option:: --num-iterations <INT>

    Maximum number of iterations (Default: 10)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by |spi| scripts... <spider-options>`
    #. :ref:`Options shared by MPI-enabled scripts... <mpi-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`
    #. :mod:`See prepare_volume for more options... <arachnid.pyspider.prepare_volume>`
    #. :mod:`See reconstruct for more options... <arachnid.pyspider.reconstruct>`
    #. :mod:`See classify for more options... <arachnid.pyspider.classify>`
    #. :mod:`See align for more options... <arachnid.pyspider.align>`

http://stackoverflow.com/questions/13101780/representing-a-simple-function-with-0th-bessel-function
.. Created on Jul 15, 2011
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.metadata import spider_params, format, format_utility
from ..core.parallel import mpi_utility
from ..core.image import analysis
from ..core.spider import spider
import reconstruct, prepare_volume, align, refine
import logging, numpy, os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, alignment, refine_index=-1, output="", leave_out=0, **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
    
    files : list
            List of input filenames
    alignment : str
               Input alignment filename
    refine_index : int
                   Starting iteration for refinement (-1, means start at end)
    output : str
             Output filename for refinement
    extra : dict
            Unused keyword arguments
    '''
    
    from ..core.parallel import openmp
    openmp.set_thread_count(extra['thread_count'])
    
    #min_resolution
    spi = spider.open_session(files, **extra)
    alignment, refine_index = refine.get_refinement_start(spi.replace_ext(alignment), refine_index, spi.replace_ext(output))
    #  1     2    3    4     5   6  7  8   9    10        11    12   13 14 15      16          17       18
    #"epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus"
    if leave_out > 0:
        if mpi_utility.is_root(**extra):
            selection = numpy.ones(len(alignment), dtype=numpy.bool)
            leave_out = int(leave_out*len(alignment))
            index = numpy.arange(len(alignment), dtype=numpy.int)
            numpy.random.shuffle(index)
            selection[index[:leave_out]]=0
        else: selection = None
        selection = mpi_utility.broadcast(selection, **extra)
    else: selection = None
    alignvals = format.read_array_mpi(spi.replace_ext(alignment), sort_column=17, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus".split(','), **extra)
    assert(alignvals.shape[1]==18)
    if refine_index == 0: alignvals[:, 14] = 1.0/len(alignvals)
    curr_slice = mpi_utility.mpi_slice(len(alignvals), **extra)
    extra.update(align.initalize(spi, files, alignvals[curr_slice], alignvals, **extra))
    if mpi_utility.is_root(**extra): refine.setup_log(output, [_logger])
    refine_volume(spi, alignvals, curr_slice, refine_index, output, selection=selection, **extra)
    if mpi_utility.is_root(**extra): _logger.info("Completed")
    
def refine_volume(spi, alignvals, curr_slice, refine_index, output, resolution_start=30.0, num_iterations=0, aggressive=False, fast=False, **extra):
    ''' Refine a volume for the specified number of iterations
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    alignvals : array
            Array of alignment parameters
    curr_slice : slice
                 Slice of align or selection arrays on current node
    refine_index : int
                   Starting offset in refinement
    output : str
             Output filename for refinement
    resolution_start : float
                       Starting resolution
    num_iterations : int
                     Maximum number of allowed iterations
    extra : dict
            Unused keyword arguments
    '''
    
    
    #extra.update(max_resolution = resolution_start, hp_radius=extra['pixel_diameter']*extra['apix'])
    extra.update(max_resolution = resolution_start)
    refine_name = "theta_delta,angle_range,trans_range,trans_step,apix,hp_radius,bin_factor,min_resolution".split(',')
    output_volume, resolution_start = refine.recover_volume(spi, alignvals, curr_slice, refine_index, output, resolution_start, **extra)
    res_iteration = numpy.zeros((num_iterations+1, 3))
    res_iteration[0, :]=(0, resolution_start, 0)
    resolution_file = spi.replace_ext(format_utility.add_prefix(output, 'res_refine'))
    load_state(resolution_file, res_iteration, output, **extra)
    if mpi_utility.is_root(**extra):
        res = res_iteration[refine_index-1, 1] if refine_index > 0 else resolution_start
        _logger.info("Starting refinement from iteration %d - resolution %f"%(refine_index, res))
    resolution_starta=resolution_start
    resolution_start=update_resolution(res_iteration, refine_index, resolution_start, **extra)
    if resolution_start == 0:
        _logger.error("here: %d, %"%(refine_index, resolution_starta))
        raise ValueError, "Zero resolution: %d"%resolution_start
    if extra['max_resolution']==resolution_start and refine_index>0: raise ValueError, "Unable to determine resolution for iteration before: %d"%refine_index
    
    extra.update(hp_type=3, trans_max=7, trans_range=500, trans_step=1, angle_range=0)
    extra.update(trans_range=ensure_translation_range(**extra), min_bin_factor=ensure_min_bin(**extra))
    res_iteration[0, 2] = ensure_translation_range(**extra)
    param=dict(extra)
    
    extra['cleanup_fft']=fast
    for refine_index in xrange(refine_index, num_iterations):
        extra.update(auto_update_param(res_iteration, refine_index, alignvals, **param))
        if extra['theta_delta'] == 0:
            raise ValueError, "Theta delta: %f, %f"%(resolution_start, res_iteration[refine_index-1, 1])
        if mpi_utility.is_root(**extra):
            _logger.info("Refinement started: %d. %s"%(refine_index+1, ",".join(["%s=%s"%(name, str(extra[name])) for name in refine_name])))
        
        target_bin = decimation_level(extra['target_resolution'], **param) if fast else 1.0#, extra['max_resolution']
        resolution_start = refine.refinement_step(spi, alignvals, curr_slice, output, output_volume, refine_index, target_bin=(target_bin, param), **extra)
        mpi_utility.barrier(**extra)
        resolution_start = mpi_utility.broadcast(resolution_start, **extra)
        res_iteration[refine_index] = (refine_index, resolution_start, 0)
        if mpi_utility.is_root(**extra): 
            valid_err = 0
            if extra['selection'] is not None:
                valid_err=numpy.mean(alignvals[numpy.logical_not(extra['selection']), 10])
            train_err=numpy.mean(alignvals[:, 10])
            _logger.info("Refinement finished: %d. %f -- %f, %f"%(refine_index+1, resolution_start, train_err, valid_err))
            numpy.savetxt(resolution_file, res_iteration, delimiter=",")
    mpi_utility.barrier(**extra)
    
def auto_update_param(res_iteration, refine_index, alignvals, max_resolution, overfilter, restrict_angle, **extra):
    '''
    @todo replace min_resolution with conservative filter values
    '''
    
    if refine_index > 0:
        trans_range = int(translation_range(alignvals, extra['apix'], **extra))
        extra.update(trans_range=trans_range)
    resolution_start = res_iteration[refine_index-1, 1] if refine_index > 0 else res_iteration[refine_index, 1]
    '''
    if refine_index > 1:
        resolution_start = res_iteration[refine_index-1, 1]
        num_iter_unchanged=0
        last_unchanged=0
        for i in xrange(2, refine_index-1):
            if res_iteration[i, 2] >=3: continue
            unchanged = numpy.sum( (res_iteration[:i-1, 1]-res_iteration[i, 1]) < resolution_start/60 )
            if unchanged > last_unchanged: num_iter_unchanged = num_iter_unchanged+1
            last_unchanged = unchanged
        
        
        #b = decimation_level(resolution_start*0.9, max_resolution, **extra)
        b = decimation_level(resolution_start, max_resolution, **extra)
        resolution_start = (extra['apix']*3)*b
        
        if mpi_utility.is_root(**extra):
            _logger.info("Number of unchanged iterations: %d (%f -> %f)"%(num_iter_unchanged, resolution_start, (extra['apix']*3)*b))
        if (trans_range/b) >= 3: num_iter_unchanged=0
        if (num_iter_unchanged+1) < 3 and extra['oversample'] > 1.0:
            extra['oversample']=1.0
        if 1 == 1:
            resolution_start *= numpy.power(0.9, int(num_iter_unchanged+1))
            #extra['hp_radius'] /= numpy.power(1.5, int(num_iter_unchanged))
        else:
            resolution_start *= 0.9
    elif refine_index > 0: resolution_start = res_iteration[refine_index-1, 1]
    else: resolution_start = res_iteration[refine_index, 1]
    '''
    
    extra.update(bin_factor=decimation_level(resolution_start, max_resolution, **extra))
    target_resolution = extra['bin_factor']*3*extra['apix']
    extra.update(theta_delta=theta_delta_est(target_resolution, **extra), target_resolution=target_resolution)
    #extra.update(bin_factor=decimation_level(resolution_start*0.9, max_resolution, **extra))
    extra.update(spider.scale_parameters(**extra))
    #extra.update(theta_delta=theta_delta_est(resolution_start, **extra))
    if refine_index > 1 and extra['theta_delta'] < restrict_angle:
        extra.update(angle_range = angular_restriction(alignvals, **extra))
    extra.update(trans_range=max(extra['trans_range'], 2), min_resolution=res_iteration[refine_index-1, 1]*overfilter)
    res_iteration[refine_index-1, 2]=extra['trans_range']
    return extra
    
def auto_update_param_old(res_iteration, refine_index, alignvals, max_resolution, overfilter, restrict_angle, **extra):
    '''
    @todo replace min_resolution with conservative filter values
    '''
    
    if refine_index > 0:
        trans_range = int(translation_range(alignvals, extra['apix'], **extra))
        extra.update(trans_range=trans_range)
    if refine_index > 1:
        resolution_start = res_iteration[refine_index-1, 1]
        num_iter_unchanged=0
        last_unchanged=0
        for i in xrange(2, refine_index-1):
            if res_iteration[i, 2] >=3: continue
            unchanged = numpy.sum( (res_iteration[:i-1, 1]-res_iteration[i, 1]) < resolution_start/60 )
            if unchanged > last_unchanged: num_iter_unchanged = num_iter_unchanged+1
            last_unchanged = unchanged
        
        
        #b = decimation_level(resolution_start*0.9, max_resolution, **extra)
        b = decimation_level(resolution_start, max_resolution, **extra)
        if mpi_utility.is_root(**extra):
            _logger.info("Number of unchanged iterations: %d (%f -> %f)"%(num_iter_unchanged, resolution_start, (extra['apix']*3)*b))
        if (trans_range/b) >= 3: num_iter_unchanged=0
        if (num_iter_unchanged+1) < 3 and extra['oversample'] > 1.0:
            extra['oversample']=1.0
        if 1 == 1:
            resolution_start *= numpy.power(0.9, int(num_iter_unchanged+1))
            #extra['hp_radius'] /= numpy.power(1.5, int(num_iter_unchanged))
        else:
            resolution_start *= 0.9
    elif refine_index > 0: resolution_start = res_iteration[refine_index-1, 1]
    else: resolution_start = res_iteration[refine_index, 1]
    
    extra.update(theta_delta=theta_delta_est(resolution_start, **extra), target_resolution=resolution_start*0.8)
    extra.update(bin_factor=decimation_level(resolution_start, max_resolution, **extra))
    #extra.update(bin_factor=decimation_level(resolution_start*0.9, max_resolution, **extra))
    extra.update(spider.scale_parameters(**extra))
    #extra.update(theta_delta=theta_delta_est(resolution_start, **extra))
    if refine_index > 1 and extra['theta_delta'] < restrict_angle:
        extra.update(angle_range = angular_restriction(alignvals, **extra))
    extra.update(trans_range=max(extra['trans_range'], 2), min_resolution=res_iteration[refine_index-1, 1]*overfilter)
    res_iteration[refine_index-1, 2]=extra['trans_range']
    return extra
    
def theta_delta_est(resolution, apix, pixel_diameter, trans_range, theta_delta, trans_max=8, oversample=1.0, **extra):
    ''' Angular sampling rate
    
    :Parameters:
    
    resolution : float
                 Current resolution of the volume
    apix : float
           Pixel size
    pixel_diameter : int
                     Diameter of the particle in pixels
    trans_range : int
                  Maximum translation range
    theta_delta : float
                  Current theta delta
    extra : dict
            Unused keyword arguments
           
    :Returns:
    
    theta_delta : float
                  Angular sampling rate
    '''
    
    if int(spider.max_translation_range(**extra)/2.0) > trans_range or trans_range <= trans_max or 1 == 1:
        theta_delta = numpy.rad2deg( numpy.arctan( resolution / (pixel_diameter*apix) ) )/oversample
        #theta_delta = numpy.rad2deg( numpy.arctan( resolution / (pixel_diameter*apix) ) )*2
        if mpi_utility.is_root(**extra):
            _logger.info("Angular Sampling: %f -- Resolution: %f -- Size: %f"%(theta_delta, resolution, pixel_diameter*apix))
    return min(15, theta_delta)
    
def decimation_level(resolution, max_resolution, apix, min_bin_factor, window, trans_range, **extra):
    ''' Estimate the level of decimation required
    
    :Parameters:
    
    resolution : float
                 Current resolution of the volume
    apix : float
           Pixel size
    max_resolution : float
                     Maximum allowed resolution
    extra : dict
            Unused keyword arguments
           
    :Returns:
    
    decimation : int
                 Level of decimation
    '''
    
    dec =  min(max(1, min(max_resolution/(apix*3), resolution / (apix*3))), min_bin_factor)
    d = float(window)/dec + 10
    d = window/float(d)
    
    if 1 == 0 and trans_range is not None:
        d2 = min(trans_range/2.0, min_bin_factor)
        d = max(d, d2)
    return max(d, 1)

def ensure_translation_range(window, ring_last, trans_range, **extra):
    ''' Ensure a valid translation range
    
    :Pararameters:
    
    window : int
             Size of the window
    ring_last : int
                Last alignment ring
    trans_range : int
                  Current translation range
    
    :Returns:
    
    trans_range : int
                  Proper translation range
    '''
    
    if (window/2 - ring_last - trans_range) < 3:
        #_logger.warn("%d = %d, %d, %d"%((window/2 - ring_last - trans_range), window/2, ring_last, trans_range ))
        return spider.max_translation_range(window, ring_last)
    return trans_range
    
def translation_range(alignvals, apix2, **extra):
    ''' Estimate the tail of the translation distribution
    
    :Parameters:
    
    alignvals : array
                Array to estimate the translations from
    
    :Returns:
    
    trans_range : float
            New translation range
    '''
    
    t = numpy.abs(alignvals[:, 12:14].ravel())/apix2
    mtrans = numpy.median(t)
    #strans = numpy.std(t)
    strans = analysis.robust_sigma(t)
    trans_range= int(mtrans+strans*4)
    if mpi_utility.is_root(**extra):
        _logger.info("Translation Range: %f -- Median: %f -- STD: %f"%(trans_range, mtrans, strans))
    return trans_range

def angular_restriction(alignvals, theta_delta, **extra):
    ''' Determine the angular restriction to apply to each projection
    
    :Parameters:
    
    theta_delta : float
                  Angular sample rate
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    ang : float
          Amount of angular restriction
    '''
    
    gdist = alignvals[:, 9]
    gdist = gdist[gdist > 0.0]
    if len(gdist) == 0: 
        _logger.warn("gdist has zero elements")
        return 0
    mang = numpy.median(gdist)
    sang = analysis.robust_sigma(gdist)
    if numpy.isnan(sang):
        _logger.warn("Nan for sang")
        return 0;
    #sang = numpy.std(gdist)
    gdist = mang+sang*4
    mdist = numpy.max(gdist)
    ang = min(mdist, max(gdist, 4*theta_delta))
    if mpi_utility.is_root(**extra):
        _logger.info("Angular Restriction: %f -- Median: %f -- STD: %f -- theta: %f -- max: %f"%(ang, mang, sang, theta_delta, mdist))
    if ang > 180.0: ang = 0
    return ang
    
def filter_resolution(bin_factor, apix, **extra):
    ''' Resolution for conservative filtering
    
    :Parameters:
    
    bin_factor : int
                 Level of decimation
    apix : float
           Pixel size
    extra : dict
            Unused keyword arguments
           
    :Returns:
    
    resolution : float
                 Resolution to filter
    '''
    
    return (bin_factor+1)*4*apix

def load_state(resolution_file, res_iteration, output, **extra):
    '''
    '''
    
    if mpi_utility.is_root(**extra):
        if os.path.exists(resolution_file):
            tmp = numpy.loadtxt(resolution_file, delimiter=",")
            res_iteration[:tmp.shape[0]]=tmp
    res_iteration[:] = mpi_utility.broadcast(res_iteration, **extra)

def ensure_min_bin(window, pixel_diameter, **extra):
    '''
    '''
    
    return (window-pixel_diameter)/10.0

def update_resolution(res_iteration, refine_index, resolution_start, **extra):
    '''
    '''
    
    if refine_index > 0:
        if res_iteration[refine_index-1, 1] > 0.0: 
            resolution_start=res_iteration[refine_index-1, 1]
        else:
            res_iteration[refine_index-1, 1]=resolution_start
    return resolution_start

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import OptionGroup, setup_options_from_doc
    
    if main_option:        
        bgroup = OptionGroup(parser, "Primary", "Primary options to set for input and output", group_order=0,  id=__name__)
        bgroup.add_option("-i", input_files=[],          help="List of input images or stacks named according to the SPIDER format", required_file=True, gui=dict(filetype="file-list"))
        bgroup.add_option("-o", output="",               help="Base filename for output volume and half volumes, which will be named raw_$output, raw1_$output, raw2_$output", gui=dict(filetype="save"), required_file=True)
        bgroup.add_option("-r", reference="",            help="Filename for reference with the proper pixel size", gui=dict(filetype="open"), required_file=True)
        bgroup.add_option("",   resolution_start=30.0,   help="Starting resolution for the refinement")
        bgroup.add_option("",   num_iterations=10,       help="Maximum number of iterations")
        bgroup.add_option("-a", alignment="",            help="Filename for the alignment parameters", gui=dict(filetype="open"), required_file=True)
        bgroup.add_option("",   aggressive=False,        help="Use more aggresive autorefinement")
        bgroup.add_option("",   fast=False,              help="Reconstruct smaller volumes")
        bgroup.add_option("",   leave_out=0.0,           help="Leave out a fraction of the particles for validation")
        bgroup.add_option("",   oversample=1.0,           help="Oversampling for angular search")
        bgroup.add_option("",   overfilter=1.5,           help="Overfiltering factor")
        bgroup.add_option("",   restrict_angle=3.0,         help="Angular step size at which to use estimated angular restrictions")
        pgroup.add_option_group(bgroup)
        setup_options_from_doc(parser, spider.open_session, group=pgroup)
        spider_params.setup_options(parser, pgroup, True)
        
    group = OptionGroup(parser, "Additional", "Options to customize your refinement", group_order=0,  id=__name__)
    group.add_option("",   refine_index=-1,             help="Iteration to start refinment: -1 = start at last volume based on output (if there is none, then start at beginning), 0 = start at begining, > 0 start after specific iteration")
    group.add_option("",   prep_thread=0,                help="Number of threads to use for volume preparation")
    pgroup.add_option_group(group)
    
    if main_option:
        parser.change_default(thread_count=4, log_level=3, max_ref_proj=5000)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if options.resolution_start <= 0: raise OptionValueError, "Resolution must be a positive number, ideally larger than 30"
    if options.num_iterations <= 1: raise OptionValueError, "Number of iterations must be greater than 1"
    

def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Refine the orientational assignment of a set of projections
                        
                        $ %prog image_stack_*.ter -p params.ter -r reference.ter -a align.ter -o align_0001.ter
                        
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
        supports_OMP=False,
        supports_MPI=True,
        use_version = True,
        max_filename_len = 78,
    )
def dependents(): return [reconstruct, prepare_volume, align]
if __name__ == "__main__": main()



