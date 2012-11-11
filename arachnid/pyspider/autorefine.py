''' Refine the orientational assignment of a set of projections

This |spi| batch file (`spi-refine`) performs refines the orientational assignment of a set of projections against a
continually improving reference.

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

.. option:: --bin-factor <FLOAT>
    
    Number of times to decimate params file

.. option:: --phase-flip <BOOL> 
    
    Set to True if your data stack(s) has already been phase flipped or CTF corrected (Default: False)

.. option:: -r <FILENAME>, --reference <FILENAME>
    
    Filename for reference with the proper pixel size

.. option:: -a <FILENAME>, --alignment <FILENAME> 
    
    Filename for the alignment parameters

.. option:: --refine-step <LIST>

    List of value tuples where each tuple represents a round of refinement and 
    contains a value for each parameter specified in the same order as `refine-name` 
    for each round of refinement; each round is separated with a comma; each value by 
    a colon, e.g. 15,10:0:6:1,8:0:4,1:3:1
    
Useful Options
==============

.. option:: --refine-name <LIST>

    ist of option names to change in each round of refinement, values set in `refine-step` (Default: theta-delta, angle-range, trans-range, trans-step, use-apsh)

.. option:: --refine-index <INT>
    
    Iteration to start refinment: -1 = start at last volume, 0 = start at begining, > 0 start after specific iteration (Default: -1)

.. option:: --keep-reference <BOOL>
    
    Do not change the reference - for tilt-pair analysis - second exposure

.. option:: --min-resolution <FLOAT>
    
    Minimum resolution to filter next input volume (Default: 0.0)

.. option:: --add-resolution <FLOAT>

    Additional amount to add to resolution before filtering the next reference (Default: 0.0)

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

def batch(files, alignment, refine_index=-1, output="", **extra):
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
    
    #min_resolution
    spi = spider.open_session(files, **extra)
    alignment, refine_index = refine.get_refinement_start(spi.replace_ext(alignment), refine_index, spi.replace_ext(output))
    #  1     2    3    4     5   6  7  8   9    10        11    12   13 14 15      16          17       18
    #"epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus"
    alignvals = format.read_array_mpi(spi.replace_ext(alignment), sort_column=17, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus".split(','), **extra)
    assert(alignvals.shape[1]==18)
    curr_slice = mpi_utility.mpi_slice(len(alignvals), **extra)
    extra.update(align.initalize(spi, files, alignvals[curr_slice], **extra))
    if mpi_utility.is_root(**extra): refine.setup_log(output)
    refine_volume(spi, alignvals, curr_slice, refine_index, output, **extra)
    if mpi_utility.is_root(**extra): _logger.info("Completed")
    
def refine_volume(spi, alignvals, curr_slice, refine_index, output, resolution_start=30.0, num_iterations=0, **extra):
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
    
    max_resolution = resolution_start
    refine_name = "theta_delta,angle_range,trans_range,trans_step,use_apsh,min_resolution,apix,bin_factor,window,shuffle_angles".split(',')
    param=dict(extra)
    angle_range, trans_range = None, None
    param['trans_range']=500
    param['trans_step']=1
    #theta_prev = None
    param['trans_range'] = ensure_translation_range(**param)
    output_volume = refine.recover_volume(spi, alignvals, curr_slice, refine_index, output, **extra)
    param['min_bin_factor'] = int((param['window']-param['pixel_diameter'])/10.0) # min 2 pixel translation = (2+3)*2
    
    if mpi_utility.is_root(**extra):
        resolution_file = spi.replace_ext(format_utility.add_prefix(output, 'res_refine'))
        if os.path.exists(resolution_file):
            res_iteration = numpy.zeros((num_iterations+1, 3))
            tmp = numpy.loadtxt(resolution_file, delimiter=",")
            res_iteration[:tmp.shape[0]]=tmp
            resolution_start, param['trans_range'], extra['angle_range']  = res_iteration[refine_index]
        else:
            res_iteration = numpy.zeros((num_iterations+1, 3))
        _logger.info("Starting refinement from %d iteration with resolution: %f - translation: %d - angle range: %d - min-bin: %d"%(refine_index, resolution_start, param['trans_range'], extra['angle_range'], param['min_bin_factor']))
    param['trans_range'] = mpi_utility.broadcast(param['trans_range'], **extra)
    extra['angle_range'] = mpi_utility.broadcast(extra['angle_range'] , **extra)
    resolution_start = mpi_utility.broadcast(resolution_start, **extra)
    if resolution_start <= 0.0: raise ValueError, "Resolution must be greater than 0"
    for refine_index in xrange(refine_index, num_iterations):
        extra['bin_factor'] = decimation_level(resolution_start, max_resolution, **param)
        dec_level=extra['dec_level']
        param['bin_factor']=extra['bin_factor']
        extra.update(spider_params.update_params(**param))
        extra['dec_level']=dec_level
        param['bin_factor']=1.0
        extra.update(spider.scale_parameters(**extra))
        
        extra['theta_delta'] = theta_delta_est(resolution_start, **extra)
        extra['shuffle_angles'] = False #extra['theta_delta'] == theta_prev and refine_index > 0
        extra['min_resolution'] = resolution_start #filter_resolution(**param)
        if mpi_utility.is_root(**extra):
            _logger.info("Refinement started: %d. %s"%(refine_index+1, ",".join(["%s=%s"%(name, str(extra[name])) for name in refine_name])))
        resolution_start = refine.refinement_step(spi, alignvals, curr_slice, output, output_volume, refine_index, **extra)
        mpi_utility.barrier(**extra)
        if mpi_utility.is_root(**extra): 
            _logger.info("Refinement finished: %d. %f"%(refine_index+1, resolution_start))
            angle_range = angular_restriction(alignvals, **extra)
            trans_range = min(int(translation_range(alignvals, **extra)/param['apix']), param['trans_range'])
            res_iteration[refine_index+1] = (resolution_start, trans_range, angle_range)
            numpy.savetxt(resolution_file, res_iteration, delimiter=",")
        param['trans_range'] = mpi_utility.broadcast(trans_range, **extra)
        extra['angle_range'] = mpi_utility.broadcast(angle_range, **extra)
        resolution_start = mpi_utility.broadcast(resolution_start, **extra)
        #theta_prev = extra['theta_delta']
    mpi_utility.barrier(**extra)
    
def theta_delta_est(resolution, apix, pixel_diameter, trans_range, theta_delta, bin_factor, **extra):
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
    bin_factor : float
                 Decimation factor
    extra : dict
            Unused keyword arguments
           
    :Returns:
    
    theta_delta : float
                  Angular sampling rate
    '''
    
    if int(spider.max_translation_range(**extra)/2.0) > trans_range or trans_range <= 3:
        theta_delta = numpy.rad2deg( numpy.arctan( resolution / (pixel_diameter*apix) ) ) * 2
        if mpi_utility.is_root(**extra):
            _logger.info("Angular Sampling: %f -- Resolution: %f -- Size: %f"%(theta_delta, resolution, pixel_diameter*apix))
    return min(15, theta_delta)
    
def decimation_level(resolution, max_resolution, apix, min_bin_factor, **extra):
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
    
    #half - radius - 3
    
    return min(max(1, min(max_resolution/(apix*4), resolution / (apix*4))), min_bin_factor) #*0.9
    #return min(6, resolution / ( apix * 4 ))

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
    
def translation_range(alignvals, **extra):
    ''' Estimate the tail of the translation distribution
    
    :Parameters:
    
    alignvals : array
                Array to estimate the translations from
    
    :Returns:
    
    trans_range : float
            New translation range
    '''
    
    t = numpy.abs(alignvals[:, 12:14].ravel())
    mtrans = numpy.median(t)
    #strans = numpy.std(t)
    strans = analysis.robust_sigma(t)
    trans_range= max(1, int(mtrans+strans*4))
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
    
    #if theta_delta > 7.9: return 0
    gdist = alignvals[:, 9]
    gdist = gdist[gdist > 0.0]
    mang = numpy.median(gdist)
    sang = analysis.robust_sigma(gdist)
    #sang = numpy.std(gdist)
    gdist = mang+sang*4
    ang = max(gdist, 2*theta_delta)
    if mpi_utility.is_root(**extra):
        _logger.info("Angular Restriction: %f -- Median: %f -- STD: %f"%(ang, mang, sang))
    return ( ang if theta_delta <= 7.9 else 0 )
    
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
        pgroup.add_option_group(bgroup)
        setup_options_from_doc(parser, spider.open_session, group=pgroup)
        spider_params.setup_options(parser, pgroup, True)
        
    group = OptionGroup(parser, "Additional", "Options to customize your refinement", group_order=0,  id=__name__)
    group.add_option("",   refine_index=-1,             help="Iteration to start refinment: -1 = start at last volume based on output (if there is none, then start at beginning), 0 = start at begining, > 0 start after specific iteration")
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
        supports_MPI=True,
        use_version = True,
        max_filename_len = 78,
    )
def dependents(): return [reconstruct, prepare_volume, align]
if __name__ == "__main__": main()



