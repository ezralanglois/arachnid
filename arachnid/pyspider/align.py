''' Align a set of particle windows to a reference and reconstruct the windows into a new volume

This |spi| batch file (`spi-align`) performs reference-based alignment on a stack of projections, then
reconstructs them into a volume ready for refinement.

Tips
====

 #. If the data stack has already been phase flipped, then set `--phase-flip` to True
 
 #. If the data stack is not ctf corrected, then you must specify a `defocus-file`. If the stacks are organized by defocus group or micrograph,
    then this is all that is required. However, if you are using a single stack then you must specify `stack-select` to map each particle back
    to its micrograph.
 
 #. Be aware that if the data or the reference does not match the size in the params file, it will be automatically reduced in size
    Use --bin-factor to control the decimation of the params file, and thus the reference volumes, masks and data stacks.

 #. When using MPI, :option:`home-prefix` should point to a directory on the head node, :option:`local-scratch` should point to a local directory 
    on the cluster node, :option:`shared-scratch` should point to a directory accessible to all nodes
    and all other paths should be relative to :option:`home-prefix`.

Examples
========

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Run alignment over a phase flipped stack
    
    $ spi-align image_stack_*.ter -p params.ter -o alignment_0001.ter -r ref_vol_0001.spi --phase-flip

Critical Options
================

.. program:: spi-align

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

Useful Options
==============

.. option:: --max-ref-proj <INT>
    
    Maximum number of reference projections in memory

.. option:: --use-apsh <BOOL>
    
    Set True to use AP SH instead of AP REF (trade speed for improved accuracy) (Default: False)

.. option:: --use-flip <BOOL>
    
    Use the phase flipped stack for alignment (Default: False)

Volume Projection Options
=========================

These options are passed to SPIDER's `pj_3q` command, the default values are generally fine for most experiments.

.. option:: --pj-radius <float> 
    
     Radius of sphere to compute projection, if less than one use 0.69 times the diameter of the object in pixels (Default: -1)

Angular Sampling Options
========================

These options are passed to SPIDER's `vo_ea` command, the default values are generally fine for most 
experiments. The most common parameter to change is `theta-delta`, which controls the sampling rate of the Euler
sphere.

.. option:: --theta-delta <float> 
    
    Angular step for the theta angles (Default: 15.0)

.. option:: --theta-start <float> 
    
     Start of theta angle range (Default: 0.0)

.. option:: --theta-end <float> 
    
    End of theta angle range (Default: 89.9)

.. option:: --phi-start <float> 
    
    Start of phi angle range (Default: 0.0)

.. option:: --phi-end <float> 
    
    End of phi angle range (Default: 359.9)

Alignment Options
=================

These options are passed to SPIDER's `ap_sh` or 'ap_ref' command, the default values are generally fine for most experiments.

.. option:: --angle-range <FLOAT> 
    
    Maximum allowed deviation of the Euler angles, where 0.0 means no restriction (Default: 0.0)

.. option:: --angle-threshold <FLOAT> 
    
    Record differences that exceed this threshold (Default: 1.0)

.. option:: --trans-range <INT> 
    
    Maximum allowed translation; if this value exceeds the window size, then it will lowered to the maximum possible (Default: 24)

.. option:: --trans-step <INT> 
    
    Translation step size (Default: 1)

.. option:: --first-ring <INT> 
    
    First polar ring to analyze (Default: 1)

.. option:: --ring-last <INT> 
    
     Last polar ring to analyze; if this value is zero, then it is chosen to be the radius of the particle in pixels (Default: 0)

.. option:: --ring-step <INT> 
    
    Polar ring step size (Default: 1)

.. option:: --ray-step <INT> 
    
    Step for the radial array (Default: 1)

.. option:: --test-mirror <BOOL> 
    
    If true, test the mirror position of the projection (Default: True)

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
    #. :mod:`See create_align for more options... <arachnid.pyspider.create_align>`

.. Created on Jul 15, 2011
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.metadata import spider_params, format, format_utility
from ..core.orient import orient_utility
from ..core.parallel import mpi_utility, parallel_utility
from ..core.spider import spider
import reconstruct, prepare_volume, create_align
import logging, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
    
    files : list
            List of input filenames
    output : str
             Output filename for alignment file
    extra : dict
            Unused keyword arguments
    '''
    
    spi = spider.open_session(files, **extra)
    align = create_align.create_alignment(files, sort_align=True, **extra)
    curr_slice = mpi_utility.mpi_slice(len(align), **extra)
    extra.update(initalize(spi, files, align[curr_slice], **extra))
    align_to_reference(spi, align, curr_slice, **extra)
    if mpi_utility.is_root(**extra):
        write_alignment(spi.replace_ext(output), align)
        #align2=align[numpy.argsort(align[:, 4]).reshape(align.shape[0])]
        #format.write(spi.replace_ext(output), align2, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus".split(','), format=format.spiderdoc)
    vols = reconstruct.reconstruct_classify(spi, align, curr_slice, output, **extra)
    if mpi_utility.is_root(**extra):
        res = prepare_volume.post_process(vols, spi, output, **extra)
        _logger.info("Resolution = %f"%res)
        _logger.info("Completed")
    mpi_utility.barrier(**extra)

def initalize(spi, files, align, max_ref_proj, use_flip=False, **extra):
    ''' Initialize SPIDER params, directory structure and parameters as well as cache data and phase flip
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    files : list
            List of input stack filenames
    align : array
            Array of alignment parameters
    max_ref_proj : int
                   Maximum number of reference projections to hold in memory
    use_flip : bool
               Use CTF-corrected stack for alignment
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    param : dict
            Dictionary of updated parameters 
    '''
    
    #if not os.path.exists(os.path.dirname(output)): os.makedirs(os.path.dirname(output))
    param = reconstruct.initalize(spi, files, align, **extra)
    param['reference_stack'] = spi.ms(max_ref_proj, param['window'])
    param['dala_stack'] = format_utility.add_prefix(param['cache_file'], 'dala_')
    if align.shape[1] > 15:
        defs = align[:, 17]
        udefs = numpy.unique(defs)
        offset = numpy.zeros(len(udefs), dtype=numpy.int)
        for i in xrange(offset.shape[0]):
            offset[i] = numpy.sum(udefs[i] == defs)
        param['defocus_offset'] = numpy.cumsum(offset)
        #if use_flip and param['flip_stack'] is not None:
        #    param['input_stack'] = param['flip_stack']
            #param['phase_flip']=True
    extra.update(param)
    spider.ensure_proper_parameters(extra)
    return extra

def write_alignment(output, alignvals, apix=None):
    ''' Write alignment values to a SPIDER file
    
    :Parameters:
    
    output : str
             Output filename
    alignvals : array
                Alignment values
    apix : float, optional
           Pixel size to scale translation
    '''
    
    header = "epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror"
    if apix is not None:
        tmp = alignvals[:, :15].copy()
        tmp[:, 6:8] /= apix
        tmp[:, 12:14] /= apix
    elif alignvals.shape[1] > 15:
        tmp=alignvals[numpy.argsort(alignvals[:, 4]).reshape(alignvals.shape[0])]
        header += ",micrograph,stack_id,defocus"
    else: tmp = alignvals
    format.write(output, tmp, header=header.split(','), format=format.spiderdoc)

def align_to_reference(spi, align, curr_slice, reference, max_ref_proj, use_flip, use_apsh, shuffle_angles=False, **extra):
    ''' Align a set of projections to the given reference
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    align : array
            Output array of alignment values
    curr_slice : slice
                 Slice of align or selection arrays on current node
    reference : str or spider_var
                Input filename for reference used in alignment
    max_ref_proj : int
                   Maximum number of reference projections allowed in memory
    use_flip : bool
                 Set true if the input stack is already phase flipped
    use_apsh : bool
               Set true to use AP SH rather than the faster, yet less accurate AP REF
    shuffle_angles : bool
                     Shuffle the angular distribution
    extra : dict
            Unused keyword arguments
    '''
    
    extra.update(spider_params.update_params(**extra))
    extra.update(spider.scale_parameters(**extra))
    angle_rot = format_utility.add_prefix(extra['cache_file'], "rot_")
    extra.update(prealign_input(spi, align[curr_slice], use_flip=use_flip, **extra))
    reference = spider.copy_safe(spi, reference, **extra)
    angle_cache = format_utility.add_prefix(extra['cache_file'], "angles_")
    align[curr_slice, 10] = 0.0
    prev = align[curr_slice, :3].copy() if numpy.any(align[curr_slice, 1]>0) else None
    ap_sel = spi.ap_sh if use_apsh else spi.ap_ref
    if _logger.isEnabledFor(logging.DEBUG): _logger.debug("Start alignment - %s"%mpi_utility.hostname())
    if use_small_angle_alignment(spi, align[curr_slice], **extra):
        del extra['theta_end']
        angle_doc, angle_num = spi.vo_ea(theta_end=extra['angle_range'], outputfile=angle_cache, **extra)
        if shuffle_angles:
            psi, theta, phi = numpy.random.random_integers(low=0, high=360, size=3)
            if theta > 180.0: theta -= 180.0
            angle_doc=spi.vo_ras(angle_doc, angle_num, (psi, theta, phi), outputfile=angle_rot)
        assert(angle_num <= max_ref_proj)
        if extra['test_mirror']: extra['test_mirror']=False
        if use_flip:
            if mpi_utility.is_root(**extra): _logger.info("Small angle alignment on CTF-corrected stacks - started")
            align_projections_sm(spi, ap_sel, None, align[curr_slice], reference, angle_doc, angle_num, **extra)
            if mpi_utility.is_root(**extra): _logger.info("Small angle alignment on CTF-corrected stacks - finished")
        else:
            if mpi_utility.is_root(**extra): _logger.info("Small angle alignment on raw stacks - started")
            align_projections_by_defocus_sm(spi, ap_sel, align[curr_slice], reference, angle_doc, angle_num, **extra)
            if mpi_utility.is_root(**extra): _logger.info("Small angle alignment on raw stacks - finished")
    else:
        #angle_set, angle_num = spider.angle_split(spi, max_ref_proj, outputfile=angle_cache, **extra)
        angle_doc, angle_num = spi.vo_ea(outputfile=angle_cache, **extra)
        if shuffle_angles:
            psi, theta, phi = numpy.random.random_integers(low=0, high=360, size=3)
            if theta > 180.0: theta -= 180.0
            angle_doc=spi.vo_ras(angle_doc, angle_num, (psi, theta, phi), outputfile=angle_rot)
        angle_off = parallel_utility.partition_offsets(angle_num, int(numpy.ceil(float(angle_num)/max_ref_proj)))
        if use_flip:
            if mpi_utility.is_root(**extra): _logger.info("Alignment on CTF-corrected stacks - started")
            align_projections(spi, ap_sel, None, align[curr_slice], reference, angle_doc, angle_off, **extra)
            if mpi_utility.is_root(**extra): _logger.info("Alignment on CTF-corrected stacks - finished")
        else:
            if mpi_utility.is_root(**extra): _logger.info("Alignment on raw stacks - started")
            align_projections_by_defocus(spi, ap_sel, align[curr_slice], reference, angle_doc, angle_off, **extra)
            if mpi_utility.is_root(**extra): _logger.info("Alignment on raw stacks - finished")
    if _logger.isEnabledFor(logging.DEBUG): _logger.debug("End alignment - %s"%mpi_utility.hostname())
    align[curr_slice, 8] = angle_num
    align[curr_slice, 6:8] *= extra['apix']
    align[curr_slice, 12:14] *= extra['apix']
    if prev is not None:
        align[curr_slice, 9] = orient_utility.euler_geodesic_distance(prev, align[curr_slice, :3])
    if mpi_utility.is_root(**extra): _logger.info("Garther alignment to root - started")
    mpi_utility.gather_array(align, align[curr_slice], **extra)
    if mpi_utility.is_root(**extra): _logger.info("Garther alignment to root - finished")

def align_projections_sm(spi, ap_sel, align, reference, angle_doc, angle_num, offset=0, cache_file=None, input_stack=None, reference_stack=None, **extra):
    ''' Align a set of projections to the given reference by reprojecting only a partial volume
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    ap_sel : method
             Selected alignment method
    align : array
            Output array of alignment values
    reference : str or spider_var
                Input filename for reference used in alignment
    angle_doc : str
                Input filename for list of Euler angles
    angle_num: int
               Number of angles in the angle document
    offset : int
             Offset into the stack
    cache_file : str
                 Input filename for the cache name and path
    input_stack : str
                  Input filename for projection stack
    reference_stack : spider_var
                      Reference to incore SPIDER memory holding reference projections
    extra : dict
            Unused keyword arguments
    '''
    
    angle_rot = format_utility.add_prefix(cache_file, "rot_")
    tmp_align = format_utility.add_prefix(cache_file, "align_")
    for i in xrange(len(align)):
        spi.vo_ras(angle_doc, angle_num, (-align[i, 2], -align[i, 1], -align[i, 0]), outputfile=angle_rot)
        spi.pj_3q(reference, angle_rot, (1, angle_num), outputfile=reference_stack, **extra)
        ap_sel(input_stack, (offset+1,offset+1), reference_stack, angle_num, ring_file=cache_file, refangles=angle_doc, outputfile=tmp_align, **extra)
        vals = numpy.asarray(format.read(spi.replace_ext(tmp_align), numeric=True, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror".split(',')))
        align[i, :4] = vals[0, :4]
        align[i, 5:15] = vals[0, 5:]
        offset += 1

def align_projections_by_defocus_sm(spi, ap_sel, align, reference, angle_doc, angle_num, defocus_offset, **extra):
    ''' Align a set of projections to the given CTF-corrected reference by reprojecting only a partial volume
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    ap_sel : method
             Selected alignment method
    align : array
            Output array of alignment values
    reference : str or spider_var
                Input filename for reference used in alignment
    angle_doc : str
                Input filename for list of Euler angles
    angle_num : int
                Number of angles in the angle document
    defocus_offset : array
                     Array of offsets in the alignment file
    extra : dict
            Unused keyword arguments
    '''
    
    reference = spi.ft(reference)
    proj_beg = 1
    ctf_volume = None
    dreference = None
    for proj_end in defocus_offset:
        ctf = spi.tf_c3(float(align[proj_beg-1, 17]), **extra)      # Generate contrast transfer function
        ctf_volume = spi.mu(reference, ctf, outputfile=ctf_volume)  # Multiply volume by the CTF
        dreference = spi.ft(ctf_volume, outputfile=dreference)
        align_projections_sm(spi, ap_sel, align[proj_beg-1:proj_end], dreference, angle_doc, angle_num, proj_beg-1, **extra)
        proj_beg = proj_end
    
def align_projections_by_defocus(spi, ap_sel, align, reference, angle_doc, angle_rng, defocus_offset, **extra):
    ''' Align a set of projections to the given CTF-corrected reference
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    ap_sel : method
             Selected alignment method
    align : array
            Output array of alignment values
    reference : str or spider_var
                Input filename for reference used in alignment
    angle_doc : str
                Input filename for list of Euler angles
    angle_rng : array
                Offsets for Euler angle list
    defocus_offset : array
                     Array of offsets in the alignment file
    extra : dict
            Unused keyword arguments
    '''
    
    reference = spi.ft(reference)
    proj_beg = 1
    ctf_volume = None
    dreference = None
    for proj_end in defocus_offset:
        ctf = spi.tf_c3(float(align[proj_beg-1, 17]), **extra)      # Generate contrast transfer function
        ctf_volume = spi.mu(reference, ctf, outputfile=ctf_volume)  # Multiply volume by the CTF
        dreference = spi.ft(ctf_volume, outputfile=dreference)
        align_projections(spi, ap_sel, (proj_beg, proj_end), align[proj_beg-1:proj_end], dreference, angle_doc, angle_rng, **extra)
        proj_beg = proj_end

def align_projections(spi, ap_sel, inputselect, align, reference, angle_doc, angle_rng, cache_file, input_stack, reference_stack, **extra):
    ''' Align a set of projections to the given reference
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    ap_sel : method
             Selected alignment method
    inputselect : str or tuple
                  Selection file or range for input projection stack
    align : array
            Output array of alignment values
    reference : str or spider_var
                Input filename for reference used in alignment
    angle_doc : str
                Input filename for list of Euler angles
    angle_rng : array
                Offsets for Euler angle list
    cache_file : str
                 Input filename for the cache name and path
    input_stack : str
                  Input filename for projection stack
    reference_stack : spider_var
                      Reference to incore SPIDER memory holding reference projections
    extra : dict
            Unused keyword arguments
    '''
    
    ref_offset = 0
    tmp_align = format_utility.add_prefix(cache_file, "align_")
    for i in xrange(1, angle_rng.shape[0]):
        angle_num = (angle_rng[i]-angle_rng[i-1])
        spi.pj_3q(reference, angle_doc, (angle_rng[i-1]+1, angle_rng[i]), outputfile=reference_stack, **extra)
        ap_sel(input_stack, inputselect, reference_stack, angle_num, ring_file=cache_file, refangles=angle_doc, outputfile=tmp_align, **extra)
        # 1     2    3     4     5   6 7   8   9      10      11    12  13 14 15
        #epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror
        vals = numpy.asarray(format.read(spi.replace_ext(tmp_align), numeric=True, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror".split(',')))
        assert(vals.shape[1]==15)
        sel = numpy.abs(vals[:, 10]) > numpy.abs(align[:, 10])
        align[sel, :4] = vals[sel, :4]
        align[sel, 5:15] = vals[sel, 5:]
        align[sel, 3] += ref_offset
        ref_offset += angle_num
    
def use_small_angle_alignment(spi, curr_slice, theta_end, angle_range=0, **extra):
    ''' Test if small angle refinement should be used
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    curr_slice : array
                 Current alignment
    theta_end : float
                Range of theta to search
    angle_range : float
                  Amount of angular restriction in search
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    val : bool
          True if small angle refinement will be faster 
    '''
    
    if angle_range > 0 and angle_range < theta_end:
        full_count = spider.angle_count(theta_end=theta_end, **extra)
        part_count = spider.angle_count(theta_end=angle_range, **extra)*len(curr_slice)
        return part_count < full_count
    return False

def prealign_input(spi, align, input_stack, use_flip, flip_stack, dala_stack, inputangles, cache_file, **extra):
    ''' Select and pre align the proper input stack
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    align : array
            Alignment values
    input_stack : str
                  Local input stack of projections
    use_flip : bool
               Set true if the input stack is already phase flipped
    flip_stack : str
                 Filename for CTF-corrected stack
    dala_stack : str
                 Local aligned stack of projections
    inputangles : str
                 Document file with euler angles for each experimental projection (previous alignment)
    cache_file : str
                 Local cache file
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    input_stack : str
                  Input filename for the input stack
    '''
    
    if use_flip and flip_stack is not None: input_stack = flip_stack
    input_stack = spider.interpolate_stack(input_stack, outputfile=format_utility.add_prefix(cache_file, "data_ip_"), **extra)
    if inputangles is not None:
        align.write_alignment(spi.replace_ext(inputangles), align, extra['apix'])
        if not spider.supports_internal_rtsq(spi):
            if mpi_utility.is_root(**extra): _logger.info("Generating pre-align dala stack")
            input_stack = spi.rt_sq(input_stack, inputangles, outputfile=dala_stack)
    return dict(input_stack=input_stack)

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import setup_options_from_doc, OptionGroup
    
    if pgroup is None: pgroup=parser
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input images or stacks named according to the SPIDER format", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Base filename for output volume and half volumes, which will be named raw_$output, raw1_$output, raw2_$output", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-r", reference="",   help="Filename for reference with the proper pixel size", gui=dict(filetype="open"), required_file=True)
        setup_options_from_doc(parser, spider.open_session, group=pgroup)
        spider_params.setup_options(parser, pgroup, True)
    
    group = OptionGroup(parser, "Alignment Parameters", "Options controlling alignment", group_order=0,  id=__name__)
    group.add_option("",   max_ref_proj=300,       help="Maximum number of reference projections in memory", gui=dict(minimum=10))
    group.add_option("",   use_apsh=False,         help="Set True to use AP SH instead of AP REF (trade speed for accuracy)")
    group.add_option("",   use_flip=False,         help="Use the phase flipped stack for alignment")
    pgroup.add_option_group(group)
    
    group = OptionGroup(parser, "Other Parameters", "Options controlling alignment", group_order=0,  id=__name__)
    setup_options_from_doc(parser, 'pj_3q', 'vo_ea', 'ap_sh', classes=spider.Session, group=group)
    pgroup.add_option_group(group)
    if main_option:
        parser.change_default(thread_count=4, log_level=3)


def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if options.max_ref_proj < 1: raise OptionValueError, "Maximum number of reference projections must at least be 1, --max-ref-proj"
    spider_params.check_options(options)

def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Align a set of particle windows to a reference
                        
                        $ %prog image_stack_*.ter -p params.ter -r reference.ter -o align_0001.ter
                        
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
def dependents(): return [reconstruct, prepare_volume, create_align]
if __name__ == "__main__": main()

