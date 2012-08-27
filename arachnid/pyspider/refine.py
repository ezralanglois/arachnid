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

..option:: -r <FILENAME>, --reference <FILENAME>
    
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

from ..core.metadata import format, format_utility, spider_utility
from ..core.parallel import mpi_utility
from ..core.spider import spider
import reconstruct, prepare_volume, align
import logging, numpy, os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, aligment, refine_index=-1, output="", **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
    
    files : list
            List of input filenames
    aligment : str
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
    alignment, refine_index = get_refinement_start(spi.replace_ext(aligment), refine_index, spi.replace_ext(output))
    alignvals = format.read_array_mpi(spi.replace_ext(aligment), sort_column=17, **extra)
    curr_slice = mpi_utility.mpi_slice(len(alignvals), **extra)
    extra.update(align.initalize(spi, files, alignvals[curr_slice], **extra))
    setup_log(output)
    refine_volume(spi, alignvals, curr_slice, refine_index, output, **extra)
    if mpi_utility.is_root(**extra): _logger.info("Completed")
    
def refine_volume(spi, alignvals, curr_slice, refine_index, output, refine_step=[], refine_name=[], keep_reference=False, **extra):
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
    refine_name : list
                  List of option names to change in each round of refinement, values set in :option:`refine_step`
    refine_step : list
                  List of value tuples where each tuple represents a round of refinement and contains 
                  a value for each parameter specified in the same order as `refine-name` for each 
                  round of refinement; each round is separated with a comma; each value by a colon, 
                  e.g. 15,10:0:6:1,8:0:4,1:3:1
    keep_reference : bool
                     Keep the initial reference for each round of refinement
    extra : dict
            Unused keyword arguments
    '''
    
    for step in refine_step[:refine_index]:
        for i, s in enumerate(refine_name[:len(step)]):
            extra[s] = step[i]
    
    output_volume = spider_utility.spider_filename(format_utility.add_prefix(output, "vol_"), refine_index) 
    if refine_index > 0 and not os.path.exists(spi.replace_ext(output_volume)):
        vols = reconstruct.reconstruct_classify(spi, alignvals, curr_slice, output, **extra)
        if mpi_utility.is_root(**extra): 
            res = prepare_volume.post_process(vols, spi, output, output_volume, **extra)
            _logger.info("Refinement finished: %d. %f"%(refine_index, res))
        mpi_utility.barrier(**extra)
    
    for step in refine_step[refine_index:]:
        for i, s in enumerate(refine_name[:len(step)]):
            extra[s] = step[i]
        if refine_index > 0 and not keep_reference: 
            extra['reference'] = output_volume
            spi.iq_sync(extra['reference'])
        output = spider_utility.spider_filename(output, refine_index+1) 
        output_volume = spider_utility.spider_filename(output_volume, refine_index+1)
        if mpi_utility.is_root(**extra): 
            _logger.info("Refinement started: %d. %s"%(refine_index+1, ",".join(["%s=%s"%(name, str(extra[name])) for name in refine_name])))
        res = refine_step(spi, alignvals, curr_slice, output, output_volume, **extra)
        mpi_utility.barrier(**extra)
        if mpi_utility.is_root(**extra): 
            _logger.info("Refinement finished: %d. %f"%(refine_index+1, res))       
        refine_index += 1
    mpi_utility.barrier(**extra)

def refine_step(spi, alignvals, curr_slice, output, output_volume, input_stack, dala_stack, **extra):
    ''' Perform a single step of refinement
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    alignvals : array
                Array of alignment parameters
    curr_slice : slice
                 Slice of align or selection arrays on current node
    output : str
             Output filename for alignment file
    output_volume : str
                    Output filename for reconstruct reference used in the next round
    input_stack : str
                  Local input stack of projections
    dala_stack : str
                 Local aligned stack of projections
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    resolution : float
                 Resolution of the current reconstruction (only for root node)
    '''
    
    reconstruct.cache_local(spi, alignvals[curr_slice], input_stack=input_stack, **extra)
    if 1 == 1: # todo add parameter to do this
        tmp_align = format_utility.add_prefix(extra['cache_file'], "prvalgn_")
        tmp = alignvals[curr_slice, 6:8] / extra['apix']
        format.write(spi.replace_ext(tmp_align), tmp)
        spi.rt_sq(input_stack, tmp_align, outputfile=dala_stack)
    align.align_to_reference(spi, alignvals, curr_slice, inputangles=tmp_align, input_stack=dala_stack, **extra)
    if mpi_utility.is_root(**extra):
        alignvals[:]=alignvals[numpy.argsort(alignvals[:, 4]).reshape(alignvals.shape[0])]
        format.write(output, alignvals, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus".split(',')) 
    vols = reconstruct.reconstruct_classify(spi, alignvals, curr_slice, output, input_stack=input_stack, **extra)
    if mpi_utility.is_root(**extra): return prepare_volume.post_process(vols, spi, output, output_volume, **extra)
    return None

def setup_log(output):
    ''' Setup the refinement progress log file
    
    :Parameters:
    
    output : str
             Output filename for refinement
    '''
    
    log_file = os.path.splitext(format_utility.add_prefix(output, "progress_"))[0]+".log"
    format_utility.backup(log_file)
    ch = logging.FileHandler(log_file, mode='a')
    _logger.addHandler(ch)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    
def get_refinement_start(aligment, refine_index=-1, output="", **extra):
    ''' 
    :Parameters:

    aligment : str
               Input alignment filename
    refine_index : int
                   Starting iteration for refinement (-1, means start at end)
    output : str
             Output file to find
    
    :Returns:
    
    alignment : str
                Starting alignment file
    refine_index : int
                   Iteration of refinement to start
    '''
    
    if refine_index < 0:
        refine_index = 1
        while True:
            if not os.path.exists(spider_utility.spider_filename(output, refine_index)): break
            refine_index += 1
        refine_index -= 1
    elif refine_index > 0:
        if os.path.exists(spider_utility.spider_filename(output, refine_index)):
            raise IOError, "Specified refinement file does not exist: %s"%spider_utility.spider_filename(output, refine_index+1)
    if refine_index == 0:
        if os.path.exists(aligment):
            raise IOError, "Specified alignment file does not exist: %s"%aligment
    else: 
        aligment = spider_utility.spider_filename(output, refine_index)
    return aligment, refine_index

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import OptionGroup
    
    if pgroup is None: pgroup=parser
    if main_option:
        refine_name=['theta-delta', 'angle-range', 'trans-range', 'trans-step', 'use_apsh']
        refine_step = [(15,0,30,4,'True'),
                       (14,0,24,4),
                       (13,0,20,2),
                       (10,0,16,1),
                       (7.0,0,10,1,False),
                       (5.0,0,8),
                       (4.0,0,4),
                       (3.0,10,2),
                       (2.5,6,1),
                       (2.0,5),
                       (1.0,4),
                       (0.75),
                       (0.7),
                       (0.65,3),
                       (0.6),
                       (0.5,2),
                       ()]
        parser.add_option("-i", input_files=[],          help="List of input images or stacks named according to the SPIDER format", required_file=True, gui=dict(filetype="file-list"))
        parser.add_option("-o", output="",               help="Base filename for output volume and half volumes, which will be named raw_$output, raw1_$output, raw2_$output", gui=dict(filetype="save"), required_file=True)
        parser.add_option("-r", reference="",            help="Filename for reference with the proper pixel size", gui=dict(filetype="open"), required_file=True)
        parser.add_option("-a", alignment="",            help="Filename for the alignment parameters", gui=dict(filetype="open"), required_file=True)
        pgroup.add_option("",   refine_name=refine_name, help="List of option names to change in each round of refinement, values set in `refine-step`")
        pgroup.add_option("",   refine_step=refine_step, help="List of value tuples where each tuple represents a round of refinement and contains a value for each parameter specified in the same order as `refine-name` for each round of refinement; each round is separated with a comma; each value by a colon, e.g. 15,10:0:6:1,8:0:4,1:3:1")
    
    group = OptionGroup(parser, "Refinement Parameters", "Options controlling alignment", group_order=0,  id=__name__)
    group.add_option("",   refine_index=-1,             help="Iteration to start refinment: -1 = start at last volume based on output (if there is none, then start at beginning), 0 = start at begining, > 0 start after specific iteration")
    group.add_option("",   keep_reference=False,        help="Do not change the reference - for tilt-pair analysis - second exposure")
    group.add_option("",   min_resolution=0.0,          help="Minimum resolution to filter next input volume")
    group.add_option("",   add_resolution=0.0,          help="Additional amount to add to resolution before filtering the next reference")
    parser.add_option_group(group)
    
    if main_option:
        parser.change_default(thread_count=4, log_level=3, max_ref_proj=5000)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if len(options.refine_name) == 0: raise OptionValueError, "No refinement names specified"
    if len(options.refine_step) == 0: raise OptionValueError, "No refinement steps specified"
    for i in xrange(len(options.refine_name)):
        options.refine_name[i] = options.refine_name[i].replace('-', '_')
        if not hasattr(options, options.refine_name[i]):
            raise OptionValueError, "Refinement parameter name does not exist: %s (--refine-name)"%options.refine_name[i]
    for vals in options.refine_step:
         if len(vals) > len(options.refine_name): 
                raise OptionValueError, "Too many arguments in refinement step: %d - supports up to %d but found %d"%(i+1, len(options.refine_name), len(vals))

def main():
    #Main entry point for this script
    from ..core.app.program import run_hybrid_program
    
    run_hybrid_program(__name__,
        description = '''Refine the orientational assignment of a set of projections
                        
                        $ %prog image_stack_*.ter -p params.ter -r reference.ter -a align.ter -o align_0001.ter --refine-step 15,10,8,5,2
                        
                        http://
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                        
                        a cluster:
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nodes=`python -c "fin=open(\"machinefile\", 'r');lines=fin.readlines();print len([val for val in lines if val[0].strip() != '' and val[0].strip()[0] != '#'])"`
                        nohup mpiexec -stdin none -n $nodes -machinefile machinefile %prog -c $PWD/$0 --use-MPI < /dev/null > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=True,
        use_version = True,
        max_filename_len = 78,
    )
def dependents(): return [reconstruct, prepare_volume, align]
if __name__ == "__main__": main()



