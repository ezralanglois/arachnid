''' Create an initial alignment file

This |spi| batch file creates a default alignment file for use with refinement. This is an alternative
to running `spi-align` and allows you to go straight to refinement. 

Tips
====

 #. For a single stack, use `--stack-select` to assign the micrograph and stack_id number. This
    is especially important when assigning defocus to each projection.
 
 #. If the filename value for `--select-file` has a number before the extension (is
    a valid SPIDER filename), e.g. select_01.spi, then the filename is treated as
    a projection selection file organized by micrograph. Otherwise, it is considered
    a selection file over the full stack (for single stack) or micrograph selection
    file for multiple-stacks.

 #. Any projection with defocus 0 or not found in the defocus file will be automatically
    skipped.

Examples
========

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Create an 'empty' alignment file (i.e. rotation values 0) with defocus values from a
    # set of stacks
    
    $ spi-create-align stack_*.spi -o align.spi --defocus-file defocus_file.spi
    
Critical Options
================
    
.. program:: spi-create-align

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames containing stacks of projection images
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`
    
.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for the empty alignment file

Useful Options
==============

.. option:: -d <FILENAME>, --defocus-file <FILENAME>
    
    Filename for the defocus values for each micrograph

.. option:: --defocus-file-header <STR>
    
    Header labelling important columns in the `defocus_file` file (Default: id:0,defocus_file:1)

.. option:: -s <FILENAME>, --select-file <FILENAME>
    
    Filename for selection of projection or micrograph subset; Number before extension (e.g. select_01.spi) and it is assumed each selection is organized by micrograph

.. option:: --select-header <STR>
    
    Header labelling important columns in the `select-file` (Default: )

.. option:: --stack-select <FILENAME>
    
    Filename with micrograph and stack_id labels for a single full stack

.. option:: --stack-header <STR>
    
    Header labelling important columns in the `stack-select` (Default: id:0,stack_id:1,micrograph:3)

.. option:: --data-ext <STR>
    
    SPIDER extension for data files

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`

.. Created on Aug 24, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.app.program import run_hybrid_program
from ..core.metadata import format, spider_utility, format_utility
from ..core.image import ndimage_file
from ..core.orient import orient_utility, healpix
from ..core.util import numpy_ext
import numpy, logging, glob, os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, data_ext, **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
        
        files : list
                List of input filenames
        output : str
                 Output filename for reconstructed volume
        data_ext : str
                   SPIDER extension
        extra : dict
                Unused keyword arguments
    '''
    
    if len(files) == 1:
        try:
            alignvals = create_alignment_from_relion(files[0], **extra)
        except:_logger.exception("here")
        else: 
            format.write(output+data_ext, alignvals, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus".split(','), format=format.spiderdoc) 
            return
    _logger.info("Creating alignment file from input stacks")
    alignvals = create_alignment(files, **extra)
    output = os.path.splitext(output)[0]
    if data_ext != "" and data_ext[0] != '.': output+='.'
    format.write(output+data_ext, alignvals, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus".split(','), format=format.spiderdoc) 
    #spider.alignment_header(alignvals))

def create_alignment_from_relion(star_file, apix, **extra):
    '''
    '''
    
    projections = format.read(star_file, numeric=True)
    _logger.info("Creating alignment file from relion star file")
    align = numpy.zeros((len(projections), 18))
    for i in xrange(len(projections)):
        projection = projections[i]
        stack_file, stack_id = spider_utility.relion_file(projection.rlnImageName)
        psi, dx, dy = orient_utility.align_param_3D_to_2D_simple(projection.rlnAnglePsi, projection.rlnOriginX, projection.rlnOriginY)
        align[i, 1] = projection.rlnAngleTilt
        align[i, 2] = projection.rlnAngleRot
        align[i, 3] = healpix.ang2pix(3, numpy.deg2rad(projection.rlnAngleTilt), numpy.deg2rad(projection.rlnAngleRot))
        align[i, 4] = i+1
        align[i, 5] = psi
        align[i, 6] = dx*apix
        align[i, 7] = dy*apix
        align[i, 8] = 1
        align[i, 15] = spider_utility.spider_id(stack_file)
        align[i, 16] = stack_id
        align[i, 17] = projection.rlnDefocusU
    return align
        

def create_alignment(files, sort_align=False, random_subset=0, min_defocus=0, max_defocus=0, **extra):
    ''' Create empty (unaligned) alignment array
    
    :Parameters:
    
    files : list
            List of input files
    sort_align : bool
                 Sort the alignment file by defocus
    random_subset : int
                    Size of random subset
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    align : array
            2D label array (nx18)
    '''
    
    label = create_stack_label(files, **extra)
    defocus_file = read_defocus_file(**extra)
    label = select_subset(files, label, **extra)
    _logger.info("Selecting %d projections in alignment file"%len(label))
    align = numpy.zeros((len(label), 18))
    align[:, 4] = label[:, 0]
    align[:, 15] = label[:, 1]
    align[:, 16] = label[:, 2]
    if len(defocus_file) > 0:
        align[:, 17] = [defocus_file.get(l[1], 0) for l in label]
        align = align[numpy.logical_and(align[:, 17] > min_defocus, align[:, 17] < max_defocus)]
        if sort_align: align[:] = align[numpy.argsort(align[:, 17])].squeeze()
    if random_subset > 0:
        idx = numpy_ext.choice(numpy.arange(0, len(align)), random_subset, False)
        align = align[idx]
    return align
    
def select_subset(files, label, select_file, select_header="", **extra):
    ''' Select a subset of the label using the specified selection file
    
    :Parameters:
    
    files : list
            List of input files
    label : array
            Array of projection labels
    select_file : str
                  Selection file containing selected ids
    select_header : str
                    Header for the `select_file` file
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    label : array
            2D label array (nx3) subset
    '''
    
    if select_file == "": return label
    if spider_utility.is_spider_filename(select_file):
        _logger.info("Assuming projection selection file: --select-file %s"%select_file)
        mics = numpy.unique(label[:, 1]).astype(numpy.int)
        index=0
        for m in mics:
            try:
                sel = format.read(select_file, spiderid=m, header=select_header, numeric=True)
            except: continue
            label[index:index+len(sel), 1] = m
            label[index:index+len(sel), 2] = [s.id for s in sel]
            index+=len(sel)
        label=label[:index]
        '''
        select = {}
        for m in mics:
            try: sel = format.read(select_file, spiderid=m, header=select_header, numeric=True)
            except: select[m] = set()
            else: 
                try: select[m] = set([int(s.id) for s in sel])
                except: raise ValueError, "Cannot find column labelled as `id` in `--select-file` %s, please use `--select-header` to label this column"%select_file
        k=0
        for i in xrange(len(label)):
            if int(label[i, 2]) in select[int(label[i, 1])]:
                label[k]=label[i]
                k+=1
        label = label[:k]
        '''
    elif len(files) == 1:
        _logger.info("Assuming micrograph selection file: --select-file %s"%select_file)
        select = format.read(select_file, header=select_header, numeric=True)
        select, header = format_utility.tuple2numpy(select)
        try: id = header.index('id')
        except: raise ValueError, "Cannot find column labelled as `id` in `--select-file` %s, please use `--select-header` to label this column"%select_file
        else: select = set(select[:, id])
        k=0
        for i in xrange(len(label)):
            if int(label[i, 1]) in select:
                label[k]=label[i]
                k+=1
        label = label[:k]
        _logger.info("Selected %d micrographs"%(len(numpy.unique(label[:, 1]))))
    else:
        _logger.info("Assuming full stack selection file: --select-file %s"%select_file)
        select = format.read(select_file, header=select_header, numeric=True)
        select, header = format_utility.tuple2numpy(select)
        try: select = select[:, header.index('id')]-1
        except: raise ValueError, "Cannot find column labelled as `id` in `--select-file` %s, please use `--select-header` to label this column"%select_file
        label = label[select].squeeze()
    return label

def create_stack_label(files, stack_select, stack_header, **extra):
    ''' Create a label array from a list of stacks and/or `stack_select` file
    
    :Parameters:
    
    files : list
            List of input files
    stack_select : str
                   Selection file containing micrograph and stack number
    stack_header : str
                   Header for the `stack_select` file
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    label : array
            2D label array (nx3)
    '''
    
    total = ndimage_file.count_images(files)
    label = numpy.zeros((total, 3), dtype=numpy.int)
    beg, end = 0, 0
    for filename in files:
        cnt = ndimage_file.count_images(filename)
        end += cnt
        label[beg:end, 1] = spider_utility.spider_id(filename)
        label[beg:end, 2] = numpy.arange(1, cnt+1)
        beg = end
    if len(files) == 1 and stack_select != "":
        sel = format.read(stack_select, header=stack_header, numeric=True)
        sel, header = format_utility.tuple2numpy(sel)
        try:
            label[:, 0] = sel[:, header.index('id')]
        except:
            raise ValueError, "Cannot find column labelled as `id` in `--stack-select` %s, please use `--stack-header` to label this column"%stack_select
        try:
            label[:, 1] = sel[:, header.index('micrograph')]
        except:
            raise ValueError, "Cannot find column labelled as `micrograph` in `--stack-select` %s, please use `--stack-header` to label this column"%stack_select
        try:
            label[:, 2] = sel[:, header.index('stack_id')]
        except:
            raise ValueError, "Cannot find column labelled as `stack_id` in `--stack-select` %s, please use `--stack-header` to label this column"%stack_select
    else:
        label[:, 0] = numpy.arange(1, len(label)+1)
    return label

def read_defocus_file(defocus_file="", defocus_file_header="id:0,defocus:1", data_ext="", **extra):
    '''Parsing a selection that maps each micrograph to its defocus
    
    .. Order=-1
    
    :Parameters:
    
    defocus_file : str
               Defocus selection file
    defocus_file_header : str
                     Header describing SPIDER document file `defocus_file`
    data_ext : str
               SPIDER project extension for data file
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    defocus_file : dict
              Dictionary of stacks with corresponding defocus_file
    '''
    
    if defocus_file == "": return {}
    if not os.path.exists(defocus_file):
        if data_ext == "":
            f = glob.glob(defocus_file+".???")
            if len(f) == 1: defocus_file = f[0]
            else: raise ValueError, "Defocus file requires extension: %s"%str(f)
        else: defocus_file += "."+data_ext
    defocus_file_dict = format.read(defocus_file, header=defocus_file_header, numeric=True)
    defocus_file_dict = format_utility.map_object_list(defocus_file_dict)
    for key in defocus_file_dict.iterkeys():
        try:
            defocus_file_dict[key] = defocus_file_dict[key].defocus
        except:
            _logger.error("Unexpected defocus_file error: %s - %s - %s"%(str(key), str(defocus_file_dict[key]._fields), str(defocus_file_header)))
            raise
    return defocus_file_dict

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    #from ..core.app.settings import setup_options_from_doc, OptionGroup
    
    if pgroup is None: pgroup=parser
    if main_option:
        pgroup.add_option("-i", input_files=[],              help="List of input images or stacks named according to the SPIDER format", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",                   help="Base filename for output volume and half volumes, which will be named raw_$output, raw1_$output, raw2_$output", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("",   data_ext="spi",              help="SPIDER extension for data files")
        pgroup.add_option("",   apix=0.0,                      help="Pixel size")
    pgroup.add_option("-d", defocus_file ="",                help="Filename for the defocus_file values for each micrograph", gui=dict(filetype="open"), required_file=False)
    pgroup.add_option("",   defocus_file_header="id:0,defocus:1", help="Header labelling important columns in the `defocus_file` file")
    pgroup.add_option("-s", select_file ="",                 help="Filename for selection of projection or micrograph subset; Number before extension (e.g. select_01.spi) and it is assumed each selection is organized by micrograph", gui=dict(filetype="open"), required_file=False)
    pgroup.add_option("",   select_header="",                help="Header labelling important columns in the `select-file`")
    pgroup.add_option("",   stack_select ="",                help="Filename with micrograph and stack_id labels for a single full stack", gui=dict(filetype="open"), required_file=False)
    pgroup.add_option("",   stack_header="id:0,stack_id:1,micrograph:3", help="Header labelling important columns in the defocus_file file")
    pgroup.add_option("",   min_defocus=14000, help="Minimum defocus")
    pgroup.add_option("",   max_defocus=50000, help="Maximum defocus")
    pgroup.add_option("",   random_subset=0,                help="Set of random subset of the given size")
    
    if main_option:
        parser.change_default(thread_count=4, log_level=3)
    
def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    if main_option:
        if not spider_utility.test_valid_spider_input(options.input_files):
            raise OptionValueError, "Multiple input files must have numeric suffix, e.g. vol0001.spi"

def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Create an initial alignment file
                        
                        $ %prog image_stack_*.ter -d defocus_file.ter -o align.ter
                        
                        http://
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=False,
        use_version=False,
        max_filename_len=0,
    )
def dependents(): return []
if __name__ == "__main__": main()



