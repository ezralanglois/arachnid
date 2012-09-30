''' Reconstruction of aligned particles into a volume

This |spi| batch file (`spi-reconstruct`) reconstructs a volume from a stack of projections and an alignment file 
using a MPI parallelized nn4 reconstruction engine or a single node SPIDER reconstruction engine.


Tips
====
 
 #. If the data stack has already been phase flipped, then set `phase_flip` to True
 
 #. Bad particles can be removed automatically using thresholding. Set `threshold` to `Overall` or `View` to remove low cross-correltion 
    projections
    
        - Set `cc-threshold` to 0, and Otzu's method will attempt to choose the best threshold
        
        - Set `cc-threshold` to a number between 0 and 1, and it will be treated as the percentage of particles to remove
        
        - Set `cc-threshold` to a number greater than 1, and it will be treated as the number of particles to remove
 
 #. Be aware that if the data or the reference does not match the size in the params file, it will be automatically reduced in size
    Use --bin-factor to control the decimation of the params file, and thus the reference volumes, masks and data stacks.
    
 #. An alignment or configuration file may be used as input to `spi-reconstruct` however, you must specify (on the command line), 
    add (for an alignment config file) or change (for a refinement config file) the `alignment` option. It is also recommended you
    change the `output` option.
    
 #. When using MPI, :option:`home-prefix` should point to a directory on the head node, :option:`local-scratch` should point to a local directory 
    on the cluster node, :option:`shared-scratch` should point to a directory accessible to all nodes
    and all other paths should be relative to :option:`home-prefix`.
    
 #. Be aware that data cached during alignment may need to be recached if you need to rerun only the reconstruction script.

Examples
========

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Reconstruct a volume using a raw data stack
    
    $ spi-reconstruct image_stack_*.ter -p params.ter -a alignment_0001.ter -o vol_0001.spi
    
    # Reconstruct a volume using ctf-corrected data stack
    
    $ spi-reconstruct image_stack_*.ter -p params.ter -a alignment_0001.ter -o vol_0001.spi --phase-flip

Critical Options
================

.. program:: spi-reconstruct

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

.. option:: -a <FILENAME>, --alignment <FILENAME> 
    
    Filename for the alignment parameters

.. option:: --phase-flip <BOOL> 
    
    Set to True if your data stack(s) has already been phase flipped or CTF corrected (Default: False)

Reconstruction Options
======================

.. option:: --npad <INT>

    Padding for the reconstruction (Default: 2)

.. option:: --sym <STR>

    Symmetry for the reconstruction (Default: c1)

.. option:: --mult-ctf <BOOL>
    
    Multiply by the CTF rather than phase flip before backprojection

.. option:: --engine <('MPI_nn4', 'BPCG', 'BP32F')>

    Type of reconstruction engine to use
    
SIRT Options
============

.. option:: --cg-radius : int

    Radius of reconstructed object, 0 means use `pixel-radius` (Default: 0)
        
.. option:: --error-limit : float

    Stopping criteria (Default: 1e-5)
        
.. option:: --chi2-limit :  float

    Stopping criteria (Default: 0.0)
        
.. option:: --iter-limit :  float

    Maximum number of iterations (Default: 20)
        
.. option:: --mode : int

    Regularization mode: (0) No regularization (1) First derivative (2) Second derivative (3) Third derivative (Default: 1)
        
.. option:: --lambda-weight : float

    Weight of regularization (Default: 2000.0)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by |spi| scripts... <spider-options>`
    #. :ref:`Options shared by MPI-enabled scripts... <mpi-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`
    #. :mod:`See resolution for more options... <arachnid.pyspider.resolution>`
    #. :mod:`See classify for more options... <arachnid.pyspider.classify>`

.. todo:: test symmetry

.. todo:: write selection files when caching locally

.. todo:: add delete cache parameter

.. Created on Aug 15, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program

from ..core.metadata import spider_params, spider_utility, format, format_utility
from ..core.parallel import mpi_utility
from ..core.image import reconstruct as reconstruct_engine, ndimage_file
from ..core.spider import spider
import resolution, classify
#import prepare_volume
import logging, os, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, aligment, **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
        
        files : list
                List of input filenames
        output : str
                 Output filename for reconstructed volume
        aligment : str
                   Input alignment filename
        extra : dict
                Unused keyword arguments
    '''
        
    spi = spider.open_session(files, **extra)
    align = format.read_array_mpi(spi.replace_ext(aligment), **extra)
    curr_slice = mpi_utility.mpi_slice(len(align), **extra)
    extra.update(initalize(spi, files, align[curr_slice], **extra))
    vols = reconstruct_classify(spi, align, curr_slice, output, **extra)
    
    if mpi_utility.is_root(**extra):
        sp = resolution.estimate_resolution(vols[1], vols[2], format_utility.add_prefix(output, 'dres_'), **extra)[0]
        _logger.info("Resolution = %f"%(extra['apix']/sp))
        _logger.info("Completed")

def initalize(spi, files, align, param_file, phase_flip=False, local_scratch="", home_prefix="", shared_scratch="", incore=False, **extra):
    ''' Initialize SPIDER params, directory structure and parameters as well as cache data and phase flip
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    files : list
            List of input stack filenames
    align : array
            Array of alignment parameters
    param_file : str
                SPIDER params file
    phase_flip : bool
                 Set to True if your data stack(s) has already been phase flipped or CTF corrected
    local_scratch : str
                    File directory on local node to copy files (optional but recommended for MPI jobs)
    home_prefix : str
                  File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)
    shared_scratch : str
                     File directory on local node to copy files (optional but recommended for MPI jobs)
    incore : bool
             Should the processed data stacks be held in in-core memory
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    param : dict
            Dictionary of updated parameters 
    '''
    
    param = spider_params.read(spi.replace_ext(param_file), extra)
    window = param['window']
    assert(window>0)
    cache = spider_utility.spider_filename('cache_',  mpi_utility.get_rank(**extra)+1, len(str(mpi_utility.get_size(**extra))))
    if home_prefix == "": 
        home_prefix = os.path.dirname(os.path.dirname(files[0]))
    if shared_scratch == "":
        param['shared_scratch'] = os.path.join(home_prefix, 'scratch', cache)
    if local_scratch =="": 
        local_scratch = os.path.join(home_prefix, 'scratch')
    if mpi_utility.is_root(**extra):
       try: os.makedirs(os.path.dirname(param['shared_scratch']))
       except: pass
    local_scratch = os.path.join(local_scratch, cache)
    param['local_scratch'] = local_scratch
    param['cache_file'] = local_scratch
    if mpi_utility.is_root(**extra):
       try: os.makedirs(os.path.dirname(param['local_scratch']))
       except: pass
    param['flip_stack'] = None
    if phase_flip: param['flip_stack'] = format_utility.add_prefix(local_scratch, 'flip_recon_')
    param['input_stack'] = spi.ms(len(align), window) if incore else format_utility.add_prefix(local_scratch, 'stack')
    
    if len(files) == 1:
        if align.shape[1] > 16:
            try:
                filename = spider_utility.file_map(files[0], align[:, 15], 0, True)
                select = align[:, 15:17]
            except IOError: filename = None
        if filename is None:
            filename = files[0]
            select = align[:, 4]
    else:
        if align.shape[1] < 17: raise ValueError, "Alignment of stacks requires alignment file with at least 17 columns: last two columns are stack_id, window_id, respectively"
        filename = spider_utility.file_map(files[0], align[:, 15], 0, True)
        select = align[:, 15:17]
    param['master_filename'] = filename
    param['master_select'] = select
    
    extra.update(param)
    cache_local(spi, align, **extra)#, filename, select, window, param['input_stack'], param['flip_stack']
    return param

def cache_local(spi, align, master_filename, master_select, window, input_stack=None, flip_stack=None, rank=0, **extra):
    ''' Distribution the data projections to scratch directories on each local node in a cluster
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    align : array
            Array of alignment parameters
    master_filename : str or dict
                      Filename or map for the input stack
    master_select : str
                    Selection array for the input stack
    window : int
             Size of the projection window
    input_stack : str
                  Input file containing of data projections
    flip_stack : str
                  Input file containing of phase-flipped data projections
    rank : int
           Rank of the executing node
    extra : dict
            Unused keyword arguments
    '''
    
    update = spider.cache_interpolate(spi, master_filename, master_select, input_stack, window, rank)
    if update and flip_stack is not None:  
        if align.shape[1] < 18: raise ValueError, "17th column of alignment file must contain defocus"
        spider.phase_flip(spi, input_stack, align[:, 17], flip_stack, window=window, **extra)

def reconstruct_classify(spi, align, curr_slice, output, **extra):
    ''' Classify a set of projections and reconstruct a volume with the given alignment values
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    align : array
            Array of alignment parameters
    curr_slice : slice
                 Slice of align or selection arrays on current node
    output : str
             Output filename
    extra : dict
            Unused keyword arguments
                 
    :Returns:
    
    vol_output : str
                 Output file tuple for two half volumes and full volume
    '''
    
    align = align.copy()
    align[:, 6:8] /= extra['apix']
    if mpi_utility.is_root(**extra):
        selection = classify.classify_projections(align, **extra)
    else: selection = 0
    selection = mpi_utility.broadcast(selection, **extra)
    if hasattr(selection, 'ndim'):
        selection = numpy.argwhere(selection)
    return reconstruct(spi, align, selection, curr_slice, output, **extra)

def reconstruct(spi, align, selection, curr_slice, output, engine=0, input_stack=None, flip_stack=None, **extra):
    ''' Reconstruct a volume with the given alignment values
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    align : array
            Array of alignment parameters
    selection : array
                Array of selections or None
    curr_slice : slice
                 Slice of align or selection arrays on current node
    output : str
             Output filename
    engine : int
             Type of reconstruction engine: (0) MPI Nearest Neighbor (1) Conjugate gradient SIRT (3) Fourier Backprojection
    input_stack : str
                  Input file containing of data projections
    flip_stack : str
                  Input file containing of phase-flipped data projections
    extra : dict
            Unused keyword arguments
                 
    :Returns:
    
    vol_output : str
                 Output file tuple for two half volumes and full volume
    '''
    
    if flip_stack is None: flip_stack = input_stack
    output_vol = [
        format_utility.add_prefix(output, 'raw'),
        format_utility.add_prefix(output, 'raw1'),
        format_utility.add_prefix(output, 'raw2')
    ]
    if engine > 0:
        return reconstruct_SNI(spi, engine, input_stack, align, selection, curr_slice, output_vol, **extra)
    else:
        return reconstruct_MPI(spi, input_stack, align, selection, curr_slice, output_vol, **extra)

def reconstruct_SNI(spi, engine, input_stack, align, selection, curr_slice, vol_output, shared_scratch, local_scratch, **extra):
    ''' Reconstruct a volume on a single node
    
    Todo: 1) split reconstructions to seprate nodes, 2) copy full stack to each node
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    engine : int
             Type of reconstruction engine
    input_stack : str
                  Input file containing of data projections
    align : array
            Array of alignment parameters
    selection : array
                Array of selections or None
    curr_slice : slice
                 Slice of align or selection arrays on current node
    shared_scratch : str
                  Filename used to construct cache files accessible to all nodes
    local_scratch : str
                   Filename used to construct cache files accessible only to the current node
    extra : dict
            Unused keyword arguments
    vol_output : str
                 Output file tuple for two half volumes and full volume
    extra : dict
            Unused keyword arguments
                 
    :Returns:
    
    vol_output : str
                 Output file tuple for two half volumes and full volume
    '''

    selectfile = format_utility.add_prefix(local_scratch, "select_recon_")
    dala_stack = format_utility.add_prefix(shared_scratch, "dala_recon_")
    align_file = format_utility.add_prefix(local_scratch, "align_")
    if mpi_utility.is_root(**extra): _logger.info("Reconstruction on a single node - started")
    if selection is not None:
        selection = format.write(spi.replace_ext(selectfile), numpy.asarray(selection[curr_slice]), header=('id', ))
    else: selectfile = None
    
    format.write(spi.replace_ext(align_file), align[curr_slice])
    spi.rt_sq(input_stack, align_file, selectfile, outputfile=dala_stack)
    mpi_utility.barrier(**extra)
    if mpi_utility.is_root(**extra):
        dalamap = spider_utility.file_map(dala_stack, mpi_utility.get_size(**extra))
        dala_stack = spider.stack(spi, dalamap, mpi_utility.get_size(**extra), format_utility.add_prefix(local_scratch, "dala_full_"))
        if selection is not None: align = align[selection]
        format.write(spi.replace_ext(align_file), align)
        if engine == 1:
            spi.bp_cg_3(dala_stack, align_file, outputfile=vol_output, **extra)
        else:
            spi.bp_32f(dala_stack, align_file, outputfile=vol_output, **extra)
    if mpi_utility.is_root(**extra): _logger.info("Reconstruction on a single node - finished")
    return vol_output

def reconstruct_MPI(spi, input_stack, align, selection, curr_slice, vol_output, local_scratch, thread_count=0, **extra):
    ''' Reconstruct a volume on multiple nodes using MPI
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    input_stack : str
                  Input file containing of data projections
    align : array
            Array of alignment parameters
    selection : array
                Array of selections or None
    curr_slice : slice
                 Slice of align or selection arrays on current node
    local_scratch : str
                   Filename used to construct cache files accessible only to the current node
    thread_count : int
                   Number of threads used in SPIDER
    extra : dict
            Unused keyword arguments
    vol_output : str
                 Output file tuple for two half volumes and full volume
    extra : dict
            Unused keyword arguments
                 
    :Returns:
    
    vol_output : str
                 Output file tuple for two half volumes and full volume
    '''
    
    if mpi_utility.is_root(**extra): _logger.info("Reconstruction on multiple nodes - started")
    selevenfile = format_utility.add_prefix(local_scratch, "seleven_recon_")
    seloddfile = format_utility.add_prefix(local_scratch, "selodd_recon_")
    dala_stack = format_utility.add_prefix(local_scratch, "dala_recon_")#even_dala_stack
    #odd_dala_stack = format_utility.add_prefix(local_scratch, "odala_recon_")
    align_file = format_utility.add_prefix(local_scratch, "align_")
    
    if selection is not None:
        sel = selection[curr_slice]
        even = sel[numpy.arange(0, len(sel), 2, dtype=numpy.int)]
        odd = sel[numpy.arange(1, len(sel), 2, dtype=numpy.int)]
        format.write(spi.replace_ext(selevenfile), numpy.asarray(even)+1, header=('id', ))
        format.write(spi.replace_ext(seloddfile),  numpy.asarray(odd)+1,  header=('id', ))
    else:
        even = numpy.arange(0, len(align[curr_slice]), 2, dtype=numpy.int)
        odd = numpy.arange(1, len(align[curr_slice]), 2, dtype=numpy.int)
        format.write(spi.replace_ext(selevenfile), numpy.asarray(even)+1, header=('id', ))
        format.write(spi.replace_ext(seloddfile), numpy.asarray(odd)+1, header=('id', ))
        
    format.write(spi.replace_ext(align_file), align[curr_slice], header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,stack_id,micrograph,defocus".split(','))
    spi.rt_sq(input_stack, align_file, outputfile=dala_stack)
    #spi.rt_sq(input_stack, align_file, seloddfile, outputfile=odd_dala_stack)
    dala_stack = spi.replace_ext(dala_stack)
    #odd_dala_stack = spi.replace_ext(odd_dala_stack)
    if thread_count > 1 or thread_count == 0: spi.md('SET MP', 1)
    
    gen1 = ndimage_file.iter_images(dala_stack, even)
    gen2 = ndimage_file.iter_images(dala_stack, odd)
    vol = reconstruct_engine.reconstruct_nn4_3(gen1, gen2, align[even], align[odd], **extra)
    if isinstance(vol, tuple):
        for i in xrange(len(vol)):
            ndimage_file.write_image(spi.replace_ext(vol_output[i]), vol[i])
    elif vol is not None:
        ndimage_file.write_image(spi.replace_ext(vol_output[mpi_utility.get_rank(**extra)]), vol)
    
    mpi_utility.barrier(**extra)
    if thread_count > 1 or thread_count == 0: spi.md('SET MP', thread_count) 
    if mpi_utility.is_root(**extra): _logger.info("Reconstruction on multiple nodes - finished")
    return vol_output

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import setup_options_from_doc, OptionGroup
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input images or stacks named according to the SPIDER format", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Base filename for output volume and half volumes, which will be named raw_$output, raw1_$output, raw2_$output", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-a", alignment="",   help="Filename for the alignment parameters", gui=dict(filetype="open"), required_file=True)
        
        setup_options_from_doc(parser, spider.open_session, group=pgroup)
        spider_params.setup_options(parser, pgroup, True)
    pgroup.add_option("",   phase_flip=False,       help="Set to True if your data stack(s) has already been phase flipped or CTF corrected")

    rgroup = OptionGroup(parser, "Reconstruction Parameters", "Options controlling reconstruction", group_order=0,  id=__name__)
    rgroup.add_option("",   npad=2,                                 help="Padding for the reconstruction", gui=dict(minimum=1))
    rgroup.add_option("",   sym=('c1',),                            help="Symmetry for the reconstruction")
    rgroup.add_option("",   mult_ctf=False,                         help="Multiply by the CTF rather than phase flip before backprojection")
    rgroup.add_option("",   engine=('MPI_nn4', 'BPCG', 'BP32F'),    help="Type of reconstruction engine to use", default=0)
    setup_options_from_doc(parser, 'bp_cg_3', classes=spider.Session, group=rgroup)
    pgroup.add_option_group(rgroup)
    
    if main_option:
        parser.change_default(thread_count=4, log_level=3)
    
def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if main_option: spider_params.check_options(options)
    if options.npad <= 1: raise OptionValueError, "Padding cannot be less than or equal to one (--npad)"
    #if options.sym <= 1: raise OptionValueError, "Padding cannot be less than or equal to one (--pad)"

def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Reconstruction of aligned particles into a volume
                        
                        $ %prog image_stack_*.ter -p params.ter -a alignment_0001.ter -o vol_0001.ter
                        
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
def dependents(): return [resolution, classify]
if __name__ == "__main__": main()

