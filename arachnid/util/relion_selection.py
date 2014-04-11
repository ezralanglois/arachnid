''' Generate RELION selection files from SPIDER selection files

This script (`ara-selrelion`) generates a RELION selection file from a list of stacks, a defocus
file and a SPIDER params file. It can also take a RELION selection file and a SPIDER selection file
and generate a new RELION selection file. Finally, it can generate a set of SPIDER selection files (by micrograph)
from a RELION selection file.

Notes
=====

 #. Input filenames: To specifiy multiple files, either use a selection file `--selection-doc sel01.spi` with a single input file mic_00001.spi or use a single star mic_*.spi to
    replace the number.
 
 #. All other filenames: Must follow the SPIDER format with a number before the extension, e.g. mic_00001.spi. Output files just require the number of digits: `--output sndc_0000.spi`
 
 #. A stack that is not found in the defocus file is skipped. If your RELION selection file is empty, then you are likly extracting the micrograph ID from the wrong 
    column, use `--defocus-header` to select the proper column.

 #. The input file determines how the script will work, if the input is a stack or set of stacks, then it will generate a relion selection file from those stacks. If
    the input file is a relion selection file, then it will generate a new set of selection files based on the extension of the output file.

 #. An output file with the star extension, will write out a RELION star file, all other extensions will write out a SPIDER selection file

Running Script
===============

.. sourcecode :: sh
    
    # Example usage - Generate a ReLion selection file from a set of stacks
    
    $ ara-selrelion win/win_*.spi --defocus-file def_avg.spi --param-file params.spi -o relion_selection.star
    
    # Example usage - Generate a ReLion selection file from a set of stacks with a stack (or micrograph or power spectra) selection file (first column holds the ID key)
    
    $ ara-selrelion win/win_*.spi --defocus-file def_avg.spi --param-file params.spi -o relion_selection.star -s select_file.spi=id:0
    
    # Example usage - Generate a ReLion selection file from a set of stacks with manual selection
    
    $ ara-selrelion win/win_*.spi --defocus-file def_avg.spi --param-file params.spi -o relion_selection.star --good-file good_00000.spi
    
    # Example usage - Generate a ReLion selection file from a set of stacks with manual selection where selection was produced by spider and does not have a header
    
    $ ara-selrelion win/win_*.spi --defocus-file def_avg.spi --param-file params.spi -o relion_selection.star --good-file good_00000.spi=id,select
    
    # Example usage - Generate a ReLion selection file from a SPIDER selection file and a Relion selection file
    
    $ ara-selrelion relion_selection_full.star --selection-file good_classavg01.spi -o relion_selection_subset.star
    
    # Example usage - Generate SPIDER selection files by micrograph from a Relion and SPIDER selection file
    
    $ ara-selrelion relion_selection_full.star --selection-file good_classavg01.spi -o good_000001.spi


Critical Options
================

.. program:: ara-selrelion

.. option:: -i <filename1,filename2>, --input-files <filename1,filename2>, filename1 filename
    
    List of filenames for the input micrographs.
    If you use the parameters `-i` or `--inputfiles` the filenames may be comma or 
    space separated on the command line; they must be comma seperated in a configuration 
    file. Note, these flags are optional for input files; the filenames must be separated 
    by spaces. For a very large number of files (>5000) use `-i "filename*"`

.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for the relion selection file (Only required if input is a stack)

.. option:: -p <FILENAME>, --param-file  <FILENAME>
    
    SPIDER parameters file (Only required when the input is a stack)
    
.. option:: -d <FILENAME>, --defocus-file <FILENAME>
    
    SPIDER defocus file (Only required when the input is a stack)
    
Selection Options
=================
    
.. option:: -s <FILENAME>, --selection-file <FILENAME>
    
    SPIDER micrograph, class selection file, or comma separated list of classes (e.g. 1,2,3) - if select file does not have proper header, then use `--selection-file filename=id` or `--selection-file filename=id,select`
    
.. option:: -g <FILENAME>, --good-file <FILENAME>
    
    SPIDER particle selection file template - if select file does not have proper header, then use `--good-file filename=id` or `--good-file filename=id,select`
    
.. option:: --tilt-pair <FILENAME>
    
    Selection file that defines pairs of particles (e.g. tilt pairs micrograph1, id1, micrograph2, id2) - outputs a tilted/untilted star files
    
.. option:: --min-defocus <float>
    
    Minimum allowed defocus
    
.. option:: --max-defocus <float>
    
    Maximum allowed defocus
    
.. option:: --view-resolution <int>
    
    Select a subset to ensure roughly even view distribution (0, default, disables this feature)
    
.. option:: --remove-missing <BOOL>
    
    Test if image file exists and if not, remove from star file

Movie-mode Options
==================

.. option:: --frame-stack-file <FILENAME>
    
    Frame stack filename used to build new relion star file for movie mode refinement. It must have a star for the frame number (e.g. --frame-stack_file frame_*_win_00001.dat)

.. option:: --frame-limit <int>
    
    Limit number of frames to use (0 means no limit)

.. option:: --reindex-file <FILENAME> 
    
    Reindex file for running movie mode on a subset of selected particles in a stack
    
More Options
============
    
.. option:: -l <str>, --defocus-header <str>
    
    Column location for micrograph id and defocus value (Default: id:0,defocus:)
    
    Example defocus file
    
    | ;             id     defocus   astig_ang   astig_mag cutoff_freq
    | 1  5       21792     29654.2     34.2511     409.274     0.20764
    | 2  5       21794     32612.5     41.0366     473.659    0.201649

.. option:: -m <INT>, --minimum-group <INT>
    
    Minimum number of particles per defocus group (regroups using the micrograph name)

.. option:: --stack-file <STR>

    Used to rename the stack portion of the image name (rlnImageName); ignored when creating a relion file

.. option:: --scale <FLOAT>
    
    Used to scale the translations in a relion file (Default: 1.0)

.. option:: --column <str>
    
    Column name in relion file for selection, e.g. rlnClassNumber to select classes (Default: rlnClassNumber)

.. option:: --test-all <str>
    
    est the normalization of all the images

.. option:: --random-subset <int>
    
    Split a relion selection file into specificed number of random subsets (0 disables)

.. option:: --downsample <float>

    Downsample the windows - create new selection file pointing to decimate stacks

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    
.. Created on Nov 27, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.app import program
from ..core.metadata import spider_utility, relion_utility, format_utility, format, spider_params, selection_utility
from ..core.image import ndimage_file, ndimage_utility, ndimage_interpolate
from ..core.parallel import parallel_utility
from ..core.orient import healpix
from ..core.image.ctf import correct as ctf_correct
import numpy
import os
import logging
import operator
import glob

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, relion2spider=False, frame_stack_file="", renormalize=0, **extra):
    '''Generate a relion selection file
    
    :Parameters:
    
    filename : tuple
               File index and input filename
    relion2spider : bool
                    If true, convert relion selection file to SPIDER
                    for SPIDER refinement
    frame_stack_file : str
                       Frame stack filename to create Relion selection file for movie mode
    extra : dict
            Unused key word arguments
    '''
    
    
    if os.path.dirname(extra['output']) != "" and not os.path.exists(os.path.dirname(extra['output'])):
        _logger.info("Creating directory: %s"%os.path.dirname(extra['output']))
        try: os.makedirs(os.path.dirname(extra['output']))
        except: pass
    
    if ndimage_file.is_readable(files[0]):
        _logger.info("Generating a relion selection file from a set of stacks")
        img = ndimage_file.read_image(files[0])
        generate_relion_selection_file(files, img, **extra)
        #relion_gui3dauto.settings
        #generate_settings(**extra)
    elif len(files) == 1 and format.get_format(files[0]) != format.star:
        generate_from_spider_alignment(files[0], **extra)
    else:
        _logger.info("Transforming %d relion selection files"%len(files))
        
        vals = []
        for f in files:
            try:
                vals.append(format.read(f, numeric=True))
            except:
                raise ValueError, "Input not an image or a selection file"
        if len(vals) > 1:
            vals = format_utility.combine(vals)
        else: vals = vals[0]
        
        # file contains
        print_stats(vals, **extra)
        if extra['test_valid']:
            count = {}
            for v in vals:
                filename, id = relion_utility.relion_file(v.rlnImageName)
                if filename not in count: 
                    try:
                        count[filename]=ndimage_file.count_images(filename)
                    except:
                        if not os.path.exists(filename): raise IOError, "%s does not exist - check your relative path"%(filename)
                        count[filename]=ndimage_file.count_images(filename)
                if id > count[filename]: raise ValueError, "%s does not have index %d in stack of size %d"%(filename, id, count[filename])
        
        if renormalize > 0:
            _logger.info("Renormalizing images to diameter of %f"%(renormalize))
            vals = renormalize_images(vals, renormalize, **extra)
            extra['output']='data_norm.star'
            select_class_subset(vals, **extra)
        elif extra['output'] != "":
            vals = select_good(vals, **extra)
            reindex_stacks(vals, **extra)
            # output file contains
            #print_stats(vals, **extra)
            
            if frame_stack_file != "":
                create_movie(vals, frame_stack_file, **extra)
            #elif relion2spider:
            #    create_refinement(vals, **extra)
            else:
                select_class_subset(vals, **extra)
    _logger.info("Completed")
    
def generate_from_spider_alignment(filename, output, param_file, image_file, **extra):
    ''' 
    '''
    
    img = ndimage_file.read_image(image_file)
    header = "rlnImageName,rlnMicrographName,rlnDefocusU,rlnVoltage,rlnSphericalAberration,rlnAmplitudeContrast,rlnGroupNumber,araOriginalrlnImageName".split(',')
    spider_params.read(param_file, extra)
    extra.update(spider_params.update_params(float(extra['window'])/img.shape[0], **extra))
    group=[]
    label=[]
    voltage, cs, ampcont=extra['voltage'], extra['cs'], extra['ampcont']
    align, aheader = format.read_alignment(filename, ndarray=True)
    
    defocus_col = aheader.index('defocus')
    mic_col = aheader.index('micrograph')
    id_col = aheader.index('id')
    part_col = aheader.index('stack_id')
    if align.shape[1] != 18: raise ValueError, "Only supports pySPIDER alignment file: %d"%align.shape[1]
    group_ids=set()
    idlen = len(str(len(align)))
    for val in align:
        defocus = val[defocus_col]
        mic = int(val[mic_col])
        part = int(val[part_col])
        id = int(val[id_col])
        if mic not in group_ids:
            group.append((defocus, numpy.sum(align[:, 17]==mic), len(label), mic))
            group_ids.add(mic)
        if spider_utility.is_spider_filename(image_file) and os.path.exists(spider_utility.spider_filename(image_file, mic)): 
            image_file = spider_utility.spider_filename(image_file, mic)
        label.append( ["%s@%s"%(str(int(id)).zfill(idlen), image_file), str(mic), defocus, voltage, cs, ampcont, len(group)-1, "%s@%s"%(str(part).zfill(idlen), str(mic))] )
    
    if len(group) == 0: raise ValueError, "No values to write out, try changing selection file"
    _logger.debug("Regrouping")
    groupmap = regroup(group, **extra)
    _logger.debug("Updating parameters")
    update_parameters(label, header, groupmap, **extra)
    
    _logger.debug("Writing out selection file")
    format.write(output, label, header=header)

def generate_relion_selection_file(files, img, output, param_file, selection_file="", good_file="", test_all=False, reindex_file="", **extra):
    ''' Generate a relion selection file for a list of stacks, defocus file and params file
    
    :Parameters:

    files : list
            List of stack files
    img : array
          Image used to query size information
    output : str
             Filename for output selection file
    param_file : str
                 Filename for input SPIDER Params file
    selection_file : str
             Filename for input optional selection file (for micrographs)
    good_file : str
                Selection file for good windows int he stack
    test_all : bool
               Test all images to ensure proper normalization
    reindex_file : bool
                   Selection file to map images in single stack
    extra : dict
            Unused key word arguments
    '''
    
    header = "rlnImageName,rlnMicrographName,rlnDefocusU,rlnVoltage,rlnSphericalAberration,rlnAmplitudeContrast,rlnGroupNumber".split(',')
    _logger.debug("Reading SPIDER params file")
    spider_params.read(param_file, extra)
    extra.update(spider_params.update_params(float(extra['window'])/img.shape[0], **extra))
    pixel_radius = int(extra['pixel_diameter']/2.0)
    if img.shape[0]%2 != 0: raise ValueError, "Relion requires even sized images"
    if img.shape[0] != img.shape[0]: raise ValueError, "Relion requires square images"
    if pixel_radius > 0:
        _logger.debug("Testing normalization")
        mask = ndimage_utility.model_disk(pixel_radius, img.shape)*-1+1
        avg = numpy.mean(img*mask)
        if numpy.allclose(0.0, avg):
            _logger.warn("Relion requires the background to be zero normalized, not %g -- for radius %d"%(avg, pixel_radius))
    
    if test_all:
        _logger.info("Testing normalization for all images - this could take some time")
        mask = ndimage_utility.model_disk(pixel_radius, img.shape)*-1+1
        mask = mask > 0
        for filename in files:
            for i, img in enumerate(ndimage_file.iter_images(filename)):
                avg = numpy.mean(img[mask]) 
                std = numpy.std(img[mask])
                if not numpy.allclose(0.0, avg): raise ValueError, "Image mean not correct: mean: %f, std: %f -- %d@%s"%(avg, std, i, filename)
                if not numpy.allclose(1.0, std): raise ValueError, "Image std not correct: mean: %f, std: %f -- %d@%s"%(avg, std, i, filename)
    
    _logger.debug("Read defocus file")
    defocus_dict = read_defocus(**extra)
    if selection_file != "": 
        if os.path.exists(selection_file):
            select = format.read(selection_file, numeric=True)
            files = selection_utility.select_file_subset(files, select)
            old = defocus_dict
            defocus_dict = {}
            for s in select:
                if s.id in old: defocus_dict[s.id]=old[s.id]
        else:
            _logger.warn("No selection file found at %s - skipping"%selection_file)
    
    #spider_params.read(param_file, extra)
    voltage, cs, ampcont=extra['voltage'], extra['cs'], extra['ampcont']
    idlen = len(str(ndimage_file.count_images(files)))
    
    tilt_pair = read_tilt_pair(**extra)
    if len(tilt_pair) > 0:
        _logger.debug("Generating tilt pair selection files")
        format.write(output, generate_selection(files, header, tilt_pair[:, :2], defocus_dict, **extra), header=header, prefix="first_")
        format.write(output, generate_selection(files, header, tilt_pair[:, 2:4], defocus_dict, **extra), header=header, prefix="second_")
        _logger.debug("Generating tilt pair selection files - finished")
    else:
        _logger.debug("Generating selection file from single stack")
        label = []
        group = []
        if len(files) == 1:
            header += ['araOriginalrlnImageName']
            group_ids=set()
            if reindex_file == "": raise ValueError, "A single stack requires --reindex-file"
            _logger.debug("Reading reindex file")
            stack_map, sheader = format.read(reindex_file, ndarray=True)
            mic_col = sheader.index('micrograph')
            id_col = sheader.index('id')
            stack_id_col = sheader.index('stack_id')
            filename=files[0]
            _logger.debug("Generating relion entries")
            for val in stack_map:
                id, mic, part = int(val[id_col]), val[mic_col], val[stack_id_col]
                if mic not in defocus_dict: 
                    _logger.warn("Skipping: %s - not in defocus file"%str(mic))
                    continue
                if mic not in group_ids:
                    group.append((defocus_dict[mic].defocus, numpy.sum(stack_map[:, mic_col]==mic), len(label), mic))
                    group_ids.add(mic)
                label.append( ["%s@%s"%(str(int(id)).zfill(idlen), filename), str(mic), defocus_dict[mic].defocus, voltage, cs, ampcont, len(group)-1, "%s@%s"%(str(part).zfill(idlen), str(mic))] )
        else:
            _logger.debug("Generating selection file from many stacks: %d"%len(files))
            for filename in files:
                mic = spider_utility.spider_id(filename)
                if good_file != "":
                    if not os.path.exists(spider_utility.spider_filename(good_file, mic)): continue
                    select_vals = format.read(good_file, spiderid=mic, numeric=True)
                    if len(select_vals) > 0 and not hasattr(select_vals[0], 'id'):
                        raise ValueError, "Error with selection file (`--good-file`) missing `id` in header, return with `--good-file filename=id` or `--good-file filename=id,select` indicating which column has the id and select"
                    if len(select_vals) > 0 and 'select' in select_vals[0]._fields:
                        select_vals = [s.id for s in select_vals if s.select > 0] if len(select_vals) > 0 and hasattr(select_vals[0], 'select') else [s.id for s in select_vals]
                else:
                    select_vals = xrange(1, ndimage_file.count_images(filename)+1)
                if mic not in defocus_dict:
                    _logger.warn("Micrograph not found in defocus file: %d -- skipping"%mic)
                    continue
                
                group.append((defocus_dict[mic].defocus, len(select_vals), len(label), mic))
                for pid in select_vals:
                    #label.append( ["%s@%s"%(str(pid).zfill(idlen), filename), filename, defocus_dict[mic].defocus, voltage, cs, ampcont, mic] )
                    label.append( ["%s@%s"%(str(pid).zfill(idlen), filename), filename, defocus_dict[mic].defocus, voltage, cs, ampcont, len(group)-1] )
        if len(group) == 0: raise ValueError, "No values to write out, try changing selection file"
        _logger.debug("Regrouping")
        groupmap = regroup(group, **extra)
        _logger.debug("Updating parameters")
        update_parameters(label, header, groupmap, **extra)
        
        _logger.debug("Writing out selection file")
        format.write(output, label, header=header)

def generate_selection(files, header, select, defocus_dict, voltage=0, cs=0, ampcont=0, id_len=0, **extra):
    ''' Generate a relion selection file for a list of stacks, defocus file and params file
    
    :Parameters:

    files : list
            List of stack files
    header : list
             List of Relion header values
    select : array
             Selection that defines tilt pairs
    defocus_dict : dict
                   Dictionary mapping micrograph ID to defocus value
    voltage : float
              Electron energy, KeV
    cs : float
         Spherical aberration, mm
    ampcont : float
              Amplitude contrast ratio
    id_len : int
             Max length of SPIDER ID
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    label : list
             List of Relion selection rows
    '''
    
    if ampcont == 0 or voltage == 0 or cs == 0: raise ValueError, "Missing SPIDER params file: voltage: %f, cs: %f, ampcont: %f"%(voltage, cs, ampcont)
    idlen = len(str(select[:, 0].max()))
    label = []
    group = []
    i=0
    filename = files[0]
    while i < len(select):
        mic = int(select[i, 0])
        select_vals = select[mic==select[:, 0], 1]
        group.append((defocus_dict[mic].defocus, len(select_vals), len(label), mic))
        filename = spider_utility.spider_filename(filename, mic, id_len)
        for pid in select_vals:
            label.append( ["%s@%s"%(str(pid).zfill(idlen), filename), filename, defocus_dict[mic].defocus, voltage, cs, ampcont, len(group)-1] )
        i+=len(select_vals)
    if len(group) == 0: raise ValueError, "No values to write out, try changing selection file"
    groupmap = regroup(group, **extra)
    update_parameters(label, header, groupmap, **extra)
    return label
    
def read_tilt_pair(tilt_pair, **extra):
    ''' Read a file that maps particle IDs of tilt pairs
    
    :Parameters:
    
    tilt_pair : str
                Filename of tilt pair selection file
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    out : array
          Tiltpair_1{Micrograph id, particle id}, Tiltpair_2{Micrograph id, particle id}
    '''
    
    if tilt_pair == "": return []
    #mic1          id        mic2         id2
    return numpy.asarray(format.read(tilt_pair, numeric=True)).astype(numpy.int)
    
def read_defocus(defocus_file, defocus_header, min_defocus, max_defocus, **extra):
    ''' Read a defocus file
    
    :Parameters:
    
    defocus_file : str
                   Filename for input defocus file
    defocus_header : str
                     Header for defocus file
    min_defocus : float
                  Minimum allowed defocus
    max_defocus : float
                  Maximum allowed defocus
    extra : dict
            Unused key word arguments
    '''
    
    if defocus_file == "": return {}
    defocus_dict = format.read(defocus_file, header=defocus_header, numeric=True)
    for i in xrange(len(defocus_dict)-1, 0, -1):
        if defocus_dict[i].defocus < min_defocus or defocus_dict[i].defocus > max_defocus:
            _logger.warn("Removing micrograph %d because defocus %f violates allowed range %f-%f "%(defocus_dict[i].id, defocus_dict[i].defocus, min_defocus, max_defocus))
            del defocus_dict[i]
    return format_utility.map_object_list(defocus_dict)
    
def build_group(data):
    ''' Build a grouping from a set of relion data
    
    :Parameters:
    
    data : list
           List of alignment parameters
    
    :Returns:
    
    group : list
            List of groups: defocus, size, offset
    '''
    
    mic_index = list(data[0]._fields).index('rlnMicrographName')
    def_index = list(data[0]._fields).index('rlnDefocusU')
    data = sorted(data, key=operator.itemgetter(mic_index))
    group = []
    defocus = data[0][def_index]
    last = spider_utility.spider_id(data[0][mic_index])
    total = 0
    selected = 0
    for d in data:
        id = spider_utility.spider_id(d[mic_index])
        if id != last:
            group.append((defocus, selected, total, last))
            defocus = data[0][def_index]
            total += selected
            selected = 0
            last = id
        selected += 1
    if selected > 0:
        group.append((defocus, selected, total, last))
    if len(group) == 1: raise ValueError, "--minimum-group may be too small for your selection file of size %d"%len(data)
    return group
    
def regroup(group, minimum_group, **extra):
    ''' Regroup micrographs by defocus
    
    :Parameters:
    
    group : list
            Micrograph grouping
    minimum_group : int
                    Minimum size of defocus group
    
    :Returns:
    
    group_map : dict
                Group mapping
    '''
    
    if minimum_group == 0: return {}
    group = numpy.asarray(group)
    assert(group.shape[1]==4)
    regroup = []
    offset = 0
    total = 0
    groupmap = {}
    try:
        idx = numpy.argsort(group[:, 0])
    except:
        _logger.error("group: %s"%str(group.shape))
        raise
    
    for i in idx:
        if total <= minimum_group:
            id = int(group[i, 3])
            groupmap[id]=offset
            regroup.append(id)
            total += group[i, 1]
        if total > minimum_group:
            offset += 1
            for id in regroup: groupmap[id]=offset
            total = 0
            regroup=[]
    _logger.info("Regrouping from %d to %d"%(len(groupmap), offset))
    return groupmap
    
def update_parameters(data, header, group_map=None, scale=1.0, stack_file="", **extra):
    ''' Update parameters in a relion selection file
    
    data : list
           List of alignment parameters
    header : list
             List of column names
    group_map : dict
                Group mapping
    scale : float
            Scale factor for translations
    stack_file : str
                 Filename for image stack
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    data : list
           Updated data list
    '''
    
    if group_map is not None and len(group_map)==0: group_map=None
    group_col = header.index('rlnMicrographName')
    group_col2 = header.index('rlnGroupNumber')
    if scale != 1.0:
        try:x_col = header.index('rlnOriginX')
        except: x_col = -1
        try:y_col = header.index('rlnOriginY')
        except: y_col = -1
    else: x_col, y_col = -1, -1
    name_col = header.index('rlnImageName') if stack_file != "" else -1
    Tuple = data[0].__class__
    
    if stack_file != "" and os.path.dirname(stack_file) != "":
        stack_files=glob.glob(os.path.dirname(stack_file))
        if len(stack_files) == 0: stack_files = stack_file #raise ValueError, "Cannot find --stack-file %s"%stack_file
        elif len(stack_files) == 1: stack_files=stack_files[0]
    else: stack_files=None
    
    if stack_files is not None and isinstance(stack_files, list):
        _logger.info("Listing possible stack files for %s"%stack_file)
        for i in xrange(len(stack_files)):
            stack_files[i] = os.path.join(stack_files[i], os.path.basename(stack_file))
            _logger.info(" - %s"%stack_files[i])
            
    
    for i in xrange(len(data)):
        vals = data[i] if not isinstance(data[i], tuple) else list(data[i])
        if group_col >= 0 and group_map is not None:
            id = spider_utility.spider_id(vals[group_col])
            try:
                vals[group_col] = spider_utility.spider_filename(vals[group_col], group_map[id])
                vals[group_col2] = spider_utility.spider_id(group_map[id])
                
            except:
                _logger.error("keys: %s"%str(group_map.keys()))
                raise
        if x_col > -1: vals[x_col]*= scale
        if y_col > -1: vals[y_col]*= scale
        if name_col > -1: 
            if isinstance(stack_files, list):
                newfile = None
                for f in stack_files:
                    f = relion_utility.relion_identifier(f, vals[name_col])
                    #print vals[name_col], '->', relion_utility.relion_file(f, True), '==', os.path.exists(relion_utility.relion_file(f, True))
                    if not os.path.exists(relion_utility.relion_file(f, True)): continue
                    newfile=f
                    break
                if newfile is None: raise ValueError, "Cannot find %s"%str(vals[name_col])
                vals[name_col] = newfile
            else:
                vals[name_col] = relion_utility.relion_identifier(stack_file, vals[name_col])
        if hasattr(Tuple, '_make'): data[i] = Tuple._make(vals)
        else: data[i] = tuple(vals)
    
    return data
    
def select_good(vals, class_file, good_file, min_defocus, max_defocus, column="rlnClassNumber", view_resolution=0, view_limit=0, remove_missing=False, **extra):
    ''' Select good particles based on selection file and defocus
    range.
    
    :Parameters:
    
    vals : list
           Entries from relion selection file
    class_file : str
                     Filename for good class selection file
    good_file : str
                Selection file for good particles organized by micrograph 
    min_defocus : float
                  Minimum allowed defocus 
    max_defocus : float
                  Maximum allowed defocus 
    column : str
             Column containing the class attribute
    view_resolution : int
                      Cull views at this resolution (0 disables)
    view_limit : int
                 Maximum number of projections per view (if 0, then use median)
    remove_missing : bool
                     Remove entries where the stack is missing
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    subset : list
             Subset of good particles
    '''
    
    
    
    old_vals = vals
    vals = []
    old_max=(1e20, -1e20)
    new_max=(1e20, -1e20)
    if len(old_vals) > 0 and not hasattr(old_vals[0], 'rlnDefocusU'):
        _logger.error("Invalid header: %s"%str(old_vals[0]._fields))
        raise ValueError, "Not a valid relion seleciton file"
    for v in old_vals:
        if v.rlnDefocusU < max_defocus and v.rlnDefocusU > min_defocus:
            vals.append(v)
            if v.rlnDefocusU  > new_max[1]: new_max = (new_max[0], v.rlnDefocusU )
            if v.rlnDefocusU  < new_max[0]: new_max = (v.rlnDefocusU, new_max[1] )
        if v.rlnDefocusU  > old_max[1]: old_max = (old_max[0], v.rlnDefocusU )
        if v.rlnDefocusU  < old_max[0]: old_max = (v.rlnDefocusU, old_max[1] )
    _logger.info("Original Defocus Range: %f, %f"%old_max)
    _logger.info("Truncated Defocus Range: %f, %f"%new_max)
    if len(vals) == 0: raise ValueError, "Nothing selected from defocus range %f - %f"%(min_defocus, max_defocus)
    
    if remove_missing:
        old_vals = vals
        vals = []
        missing=set()
        for v in old_vals:
            if not os.path.exists(relion_utility.relion_file(v.rlnImageName, True)):
                missing.add(relion_utility.relion_file(v.rlnImageName, True))
            else:
                vals.append(v)
        _logger.warn("Removed %d image stacks"%len(missing))
    
    if good_file != "":
        _logger.info("Selecting good particles from: %s"%str(good_file))
        last=None
        subset=[]
        if spider_utility.is_spider_filename(relion_utility.relion_file(vals[0].rlnImageName, True)):
            for v in vals:
                mic,pid1 = relion_utility.relion_id(v.rlnImageName)
                if mic != last:
                    try:
                        select_vals = set([s.id for s in format.read(good_file, spiderid=mic, numeric=True)])
                    except:
                        if _logger.isEnabledFor(logging.DEBUG):
                            _logger.exception("Error reading selection file")
                        select_vals=set()
                    last=mic
                if pid1 in select_vals:
                    subset.append(v)
        else:
            if 1==1:
                select_vals = set([s.id for s in format.read(good_file, numeric=True)])
                _logger.info("Unique selections: %d"%len(select_vals))
                for s in select_vals:
                    v = vals[s-1]
                    assert(relion_utility.relion_id(v.rlnImageName)[1]==s)
                    subset.append(v)
            else:
                select_vals = set([s.id for s in format.read(good_file, numeric=True)])
                for v in vals:
                    _,pid1 = relion_utility.relion_id(v.rlnImageName)
                    if pid1 in select_vals:
                        subset.append(v)
            
        _logger.info("Selected %d of %d"%(len(subset), len(vals)))
        if len(subset) == 0: raise ValueError, "Nothing selected from %s"%good_file
    else: subset=vals
    
    _logger.debug("Subset size1: %d"%len(subset))
    if class_file != "" and not isinstance(class_file, list):
        select = class_file
        vals = subset
        subset=[]
        try: select=int(select)
        except:
            if select.find(",") != -1:
                select = set([int(v) for v in select.split(',')])
            else:
                select = format.read(class_file, numeric=True)
                if len(select) > 0 and hasattr(select[0], 'select'):
                    select = set([s.id for s in select if s.select > 0])
                else:
                    select = set([s.id for s in select])
        else: select=set([select])
        _logger.info("Selecting classes: %s"%str(select))
        for v in vals:
            id = getattr(v, column)
            try: id = int(id)
            except: id = spider_utility.spider_id(id)
            if id in select: subset.append(v)
        if len(subset) == 0: raise ValueError, "No classes selected"
    
    if view_resolution > 0:
        n=healpix.res2npix(view_resolution, True, True)
        _logger.info("Culling %d views with resolution %d"%(n, view_resolution))
        ang = numpy.asarray([(v.rlnAngleTilt, v.rlnAngleRot) for v in subset])
        view = healpix.ang2pix(view_resolution, numpy.deg2rad(ang), half=True)
        assert(len(numpy.unique(view)) <= n)
        vhist = numpy.histogram(view, n)[0]
        maximum_views = numpy.median(vhist) if view_limit < 1 else view_limit
        _logger.info("Maximum of %d projections allowed per view"%(maximum_views))
        assert(vhist[1] == numpy.sum(view==1))
        assert(vhist[20] == numpy.sum(view==20))
        assert(vhist[30] == numpy.sum(view==30))
        vals = subset
        subset = []
        pvals = None
        if hasattr(vals[0], 'rlnMaxValueProbDistribution'):
            _logger.info("Choosing views with highest P-value")
            pvals = numpy.asarray([v.rlnMaxValueProbDistribution for v in vals])
            idx = numpy.argsort(pvals)[::-1]
        else:
            idx = numpy.arange(len(vals), dtype=numpy.int)
            numpy.random.shuffle(idx)
        count = numpy.zeros(n)
        for i in idx:
            if count[view[i]] < maximum_views:
                subset.append(vals[i])
                count[view[i]] += 1
        _logger.info("Reduced projections from %d to %d"%(len(vals), len(subset)))
    
    return subset

def print_stats(vals, column="rlnClassNumber", **extra):
    ''' Print the statistics of the Relion selection file
    
    :Parameters:
    
    vals : list
           Entries from relion selection file
    column : str
             Column containing the class attribute
    '''
    
    if len(vals) == 0: raise ValueError, "No values read"
    tmp = numpy.asarray(vals)
    _logger.info("# of projections: %d"%len(vals))
    # min,max number of groups/micrographs
    
    if column in vals[0]._fields:
        tmp = numpy.asarray([getattr(v, column) for v in vals])
        clazzes = numpy.unique(tmp)
        for cl in clazzes:
            _logger.info("Class: %d has %d projections"%(cl, numpy.sum(cl==tmp)))
            
def create_movie(vals, frame_stack_file, output, frame_limit=0, reindex_file="", single_stack=False, **extra):
    ''' Convert a standard relion selection file to a movie mode selection file and
    write to output file
    
    :Parameters:
    
    vals : list
           Entries from relion selection file
    frame_stack_file : str
                       Frame window stack filename
    output : str
             Output filename
    frame_limit : int
                  Maximum number of frames to include
    reindex_file : str
                   Re-indexing selection file for subsets
    extra : dict
            Unused key word arguments
    '''
    
    _logger.info("Creating movie mode relion selection file: %d"%frame_limit)
    frame_vals = []
    idlen=None
    consecutive=None if reindex_file == "" else True
    last=-1
    
    stack_filename = format_utility.add_prefix(output, 'image_stack_')
    stack_index = 0
    last_percent = -1
    for offset, v in enumerate(vals):
        percent = int(float(offset)/len(vals)*100)
        if (percent%10) == 0 and percent != last_percent:
            _logger.info("Finished %d of %d -- %f"%(offset, len(vals), percent))
            last_percent=percent
        mic,pid1 = relion_utility.relion_id(v.rlnImageName)
        if mic != last:
            frames = glob.glob(spider_utility.spider_filename(frame_stack_file, mic))
            if consecutive is None:
                avg_count = ndimage_file.count_images(relion_utility.relion_file(v.rlnImageName, True))
                frm_count = ndimage_file.count_images(frames[0])
                consecutive = avg_count != frm_count
                if consecutive:
                    _logger.info("Detected fewer particles in stack %s - %d < %d (frame < average)"%(v.rlnImageName, frm_count, avg_count))
                    if reindex_file == "":raise ValueError, "Requires selection file to reindex the relion star file"
            if consecutive:
                index = format.read(reindex_file, spiderid=mic, numeric=True)
                index = dict([(val.id, i+1) for i, val in enumerate(index)])
            last=mic
            if idlen is None:
                idlen = len(str(len(vals)*len(frames)))
        if consecutive: pid = index[pid1]
        else: pid=pid1
        frames = sorted(frames)
        if frame_limit == 0: frame_limit=len(frames)
        if len(frames) < frame_limit:
            _logger.warn("Skipping %s - too few frames: %d < %d"%(v.rlnImageName, len(frames), frame_limit))
            continue
        frames=frames[:frame_limit]
        for f in frames:
            if single_stack:
                orig_image_file = "%s@%s"%(str(pid).zfill(idlen), f)
                ndimage_file.write_image(stack_filename, ndimage_file.read_image(f, pid-1), stack_index)
                image_file = "%s@%s"%(str(stack_index+1).zfill(idlen), stack_filename)
                stack_index += 1
                additional = (v.rlnImageName, orig_image_file)
            else:
                image_file = "%s@%s"%(str(pid).zfill(idlen), f)
                additional = (v.rlnImageName, )
            frame_vals.append(v._replace(rlnImageName=image_file)+additional)
    header = list(vals[0]._fields)
    header.append('rlnParticleName')
    if single_stack:
        header.append('araOriginalrlnImageName')
    format.write(output, frame_vals, header=header)
    
def create_movie_old(vals, frame_stack_file, output, frame_limit=0, **extra):
    ''' Convert a standard relion selection file to a movie mode selection file and
    write to output file
    
    :Parameters:
    
    vals : list
           Entries from relion selection file
    frame_stack_file : str
                       Frame window stack filename
    output : str
             Output filename
    frame_limit : int
                  Maximum number of frames to include
    extra : dict
            Unused key word arguments
    '''
    
    _logger.info("Creating movie mode relion selection file: %d"%frame_limit)
    frame_vals = []
    idlen=None
    consecutive=None
    last=-1
    for v in vals:
        mic,pid1 = relion_utility.relion_id(v.rlnImageName)
        if mic != last:
            frames = glob.glob(spider_utility.spider_filename(frame_stack_file, mic))
            if consecutive is None:
                avg_count = ndimage_file.count_images(relion_utility.relion_file(v.rlnImageName, True))
                frm_count = ndimage_file.count_images(frames[0])
                consecutive = False #avg_count != frm_count
                if consecutive:
                    _logger.info("Detected fewer particles in stack %s - %d < %d"%(v.rlnImageName, frm_count, avg_count))
            last=mic
            if consecutive: pid=0
            if idlen is None:
                idlen = len(str(len(vals)*len(frames)))
        if consecutive: pid += 1
        else: pid=pid1
        frames = sorted(frames)
        if frame_limit == 0: frame_limit=len(frames)
        if len(frames) < frame_limit:
            _logger.warn("Skipping %s - too few frames: %d < %d"%(v.rlnImageName, len(frames), frame_limit))
            continue
        frames=frames[:frame_limit]
        for f in frames:
            frame_vals.append(v._replace(rlnImageName="%s@%s"%(str(pid).zfill(idlen), f))+(v.rlnImageName, ))
    header = list(vals[0]._fields)
    header.append('rlnParticleName')
    format.write(output, frame_vals, header=header)

def select_class_subset(vals, output, random_subset=0, restart=False, param_file="", reindex_file="", sort_column="", sort_rev=False, split=False, **extra):
    ''' Select a subset of classes and write a new selection file
    
    :Parameter:
    
    vals : list
           List of entries from a selection file
    output : str
             Filename for output selection file
    random_subset : int
                    Generate the specified number of random subsets
    restart : bool
              If True, then output only minimum header
    param_file : str
                 SPIDER Params file
    reindex_file : str
                   Selection file for reindexing
    sort_column : str
                  Column to sort by
    sort_rev : bool
               Reverse sorting - descending order
    split : bool
            Split a relion star file
    extra : dict
            Unused key word arguments
    
    '''
    
    
    if reindex_file != "":
        last=-1
        idlen = len(str(len(vals)))
        for i in xrange(len(vals)):
            filename,pid1 = relion_utility.relion_file(vals[i].rlnImageName)
            if last != filename:
                index = format.read(reindex_file, spiderid=filename, numeric=True)
                index = dict([(val.id, i+1) for i, val in enumerate(index)])
                last=filename
            pid = index[pid1]
            vals[i] = vals[i]._replace(rlnImageName="%s@%s"%(str(pid).zfill(idlen), filename))
    
    subset=vals
    if os.path.splitext(output)[1] == '.star':
        defocus_dict = read_defocus(**extra)
        if param_file != "":
            _logger.info("Update params values")
            spider_params.read(param_file, extra)
            cs = extra['cs']
            ampcont = extra['ampcont']
            voltage = extra['voltage']
            for i in xrange(len(subset)):
                subset[i] = subset[i]._replace(rlnSphericalAberration=cs)
                subset[i] = subset[i]._replace(rlnVoltage=voltage)
                subset[i] = subset[i]._replace(rlnAmplitudeContrast=ampcont)
            
        if len(defocus_dict) > 0:
            _logger.info("Read %d values from a defocus file"%len(defocus_dict))
            for i in xrange(len(subset)):
                mic,par = relion_utility.relion_id(subset[i].rlnImageName)
                try:
                    subset[i] = subset[i]._replace(rlnDefocusU=defocus_dict[mic].defocus)
                except:
                    _logger.warn("Not updating defocus for micrograph %d"%mic)
        if sort_column != "":
            subset.sort(key=operator.attrgetter(sort_column), reverse=sort_rev)
        #rlnSphericalAberration
        
        if split:
            if len(subset) == 0: raise ValueError, "No data selected"
            if not hasattr(subset[0], 'rlnRandomSubset'): raise ValueError, "No rlnRandomSubset column found"
            
            curr_subset=[subset[j]._make(subset[j]) for j in xrange(len(subset)) if subset[j].rlnRandomSubset==1]
            try: groupmap = regroup(build_group(curr_subset), **extra)
            except: groupmap=None
            update_parameters(curr_subset, list(curr_subset[0]._fields), groupmap, **extra)
            format.write(output, curr_subset, spiderid=1)
            
            
            curr_subset=[subset[j]._make(subset[j]) for j in xrange(len(subset)) if subset[j].rlnRandomSubset==2]
            try: groupmap = regroup(build_group(curr_subset), **extra)
            except: groupmap=None
            update_parameters(curr_subset, list(curr_subset[0]._fields), groupmap, **extra)
            format.write(output, curr_subset, spiderid=2)
            
        elif random_subset > 1:
            _logger.info("Writing %d random subsets of the selection file"%random_subset)
            index = numpy.arange(len(subset), dtype=numpy.int)
            numpy.random.shuffle(index)
            index_sets = parallel_utility.partition_array(index, random_subset)
            for i, index in enumerate(index_sets):
                curr_subset=[subset[j]._make(subset[j]) for j in index]
                try: groupmap = regroup(build_group(curr_subset), **extra)
                except: groupmap=None
                update_parameters(curr_subset, list(curr_subset[0]._fields), groupmap, **extra)
                format.write(output, curr_subset, spiderid=(i+1))
        else:
            try: groupmap = regroup(build_group(subset), **extra)
            except: groupmap=None
            update_parameters(subset, list(subset[0]._fields), groupmap, **extra)
            
            subset=downsample_images(subset, param_file=param_file, **extra)
            
            if restart:
                format.write(output, subset, header="rlnImageName,rlnMicrographName,rlnDefocusU,rlnVoltage,rlnSphericalAberration,rlnAmplitudeContrast,rlnGroupNumber".split(','))
            else:
                format.write(output, subset)
    else:
        micselect={}
        for v in subset:
            mic,par = relion_utility.relion_id(v.rlnImageName)
            if par is None:
                
                if mic not in micselect: micselect[mic]=[]
            else:
                if mic not in micselect: micselect[mic]=[]
                micselect[mic].append((par, 1))
        _logger.info("Writing SPIDER selection files for %d micrographs"%len(micselect))
        prefix=None
        if spider_utility.is_spider_filename(output):
            for mic,vals in micselect.iteritems():
                format.write(output, numpy.asarray(vals), spiderid=mic, header="id,select".split(','), format=format.spidersel)
            prefix='mic_'
        format.write(output, numpy.hstack((numpy.asarray(micselect.keys())[:, numpy.newaxis], numpy.ones(len(micselect.keys()))[:, numpy.newaxis])), header="id,select".split(','), format=format.spidersel, prefix=prefix)

def reindex_stacks(subset, reindex=False, **extra):
    ''' Restart the index of the particle number from 1.
    
    This is necessary when you crop a subset of windows based on a selection derived
    from the input relion seleciton file.
    
    :Parameters:
    
    vals : list
           List of entries from a selection file ( replaced in-place)
    reindex : bool
              If True, perform reindexing
    extra : dict
            Unused key word arguments
    '''
    
    if not reindex: return subset
    _logger.info("Re-indexing star file")
    idmap={}
    for i in xrange(len(subset)):
        filename = relion_utility.relion_file(subset[i].rlnImageName)[0]
        if filename not in idmap: idmap[filename]=0
        idmap[filename] += 1
        subset[i] = subset[i]._replace(rlnImageName=relion_utility.relion_identifier(filename, idmap[filename]))

def renormalize_images(vals, pixel_radius, apix, output, invert=False, dry_run=False, param_file="", **extra):
    ''' Renormalize a set of images in-place
    
    :Parameters:
    
    vals : list
           List of entries from a selection file
    pixel_radius : float
                   Radius of the mask in angstroms
    apix : float
           Radius of the mask in angstroms
    output : str
             Output filename for image
    extra : dict
            Unused key word arguments
    
    '''
    
    if apix == 0:
        if param_file == "": raise ValueError, "Requires --apix or --params-file"
        spider_params.read(param_file, extra) 
        apix = extra['apix']
    pixel_radius = int(float(pixel_radius)/apix)
    filename, index = relion_utility.relion_file(vals[0].rlnImageName)
    img = ndimage_file.read_image(filename, index)
    mask = ndimage_utility.model_disk(int(pixel_radius/2), img.shape)*-1+1
    assert(mask.sum()>0)
    new_vals = []
    idmap={}
    numpy.seterr(all='raise')
    for v in vals:
        filename, index = relion_utility.relion_file(v.rlnImageName)
        if spider_utility.is_spider_filename(filename):
            output = spider_utility.spider_filename(output, filename)
        if filename not in idmap: idmap[filename]=0
        if not dry_run:
            img = ndimage_file.read_image(filename, index-1)
            if img.shape[0] != mask.shape[0]:
                _logger.error("Image does not match mask (%d != %d) - %s"%(img.shape[0], mask.shape[0], filename))
            ndimage_utility.normalize_standard(img, mask, True, img)
            if invert: ndimage_utility.invert(img, img)
            ndimage_file.write_image(output, img, idmap[filename], header=dict(apix=apix))
        idmap[filename] += 1
        new_vals.append(v._replace(rlnImageName=relion_utility.relion_identifier(output, idmap[filename])))
    return new_vals

def downsample_images(vals, downsample=1.0, param_file="", phase_flip=False, apix=1.0, pixel_radius=0, mask_diameter=0, pad=1, **extra):
    ''' Downsample images in Relion selection file and update
    selection entries to point to new files.
    
    :Parameters:
    
    vals : list
           List of entries from a selection file
    downsample : float
                 Down sampling factor
    phase_flip : bool
                 Apply CTF correction by phase flipping
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    out : list
           Update list of entries from a selection file
    '''
    
    if downsample == 1.0 and not phase_flip: return vals
    if pixel_radius == 0 and mask_diameter == 0:
        if param_file == "": 
            raise ValueError, "Requires --param-file or --pixel-radius and --apix"
        extra['bin_factor']=1.0
        spider_params.read(param_file, extra)
        pixel_radius=extra['pixel_diameter']/2
        pixel_radius /= downsample
    if pixel_radius == 0:
        pixel_radius = int(mask_diameter/apix)/2
    _logger.info("Using %f angstroms as the diameter of the mask in relion"%(pixel_radius*2*apix))
    
    filename, index = relion_utility.relion_file(vals[0].rlnImageName)
    img = ndimage_file.read_image(filename)
    
    mask = None
    #ds_kernel = ndimage_interpolate.sincblackman(downsample, dtype=numpy.float32) if downsample > 1.0 else None
    ds_kernel = ndimage_interpolate.resample_offsets(img, downsample) if downsample > 1.0 else None
    filename = relion_utility.relion_file(vals[0].rlnImageName, True)
    if phase_flip:
        output = os.path.join(os.path.dirname(filename)+"_flipped_%.2f"%downsample, os.path.basename(filename))
    else:
        output = os.path.join(os.path.dirname(filename)+"_%.2f"%downsample, os.path.basename(filename))
    oindex = {}
    
    try: os.makedirs(os.path.dirname(output))
    except: pass
    
    if 'voltage' not in extra:
        extra['voltage']=vals[0].rlnVoltage
    if 'cs' not in extra:
        extra['cs']=vals[0].rlnSphericalAberration
    if 'ampcont' not in extra:
        extra['ampcont']=vals[0].rlnAmplitudeContrast
    
    if downsample > 1.0: _logger.info("Downsampling images")
    if phase_flip: _logger.info("Phase flipping images")
    _logger.info("Stack preprocessing started")
    for i in xrange(len(vals)):
        v = vals[i]
        if (i%1000) == 0:
            _logger.info("Processed %d of %d"%(i+1, len(vals)))
        filename, index = relion_utility.relion_file(v.rlnImageName)
        img = ndimage_file.read_image(filename, index-1).astype(numpy.float32)
        if phase_flip:
            ctfimg = ctf_correct.phase_flip_transfer_function(img.shape, v.rlnDefocusU, **extra)
            img = ctf_correct.correct(img, ctfimg).copy()
        if ds_kernel is not None:
            #img = ndimage_interpolate.downsample(img, downsample, ds_kernel)
            img = ndimage_interpolate.resample_fft(img, downsample, offsets=ds_kernel, pad=pad)
        if mask is None: mask = ndimage_utility.model_disk(pixel_radius, img.shape)
        ndimage_utility.normalize_standard(img, mask, out=img)
        if filename not in oindex: oindex[filename]=0
        oindex[filename] += 1
        output = spider_utility.spider_filename(output, filename)
        ndimage_file.write_image(output, img, oindex[filename]-1, header=dict(apix=apix))
        vals[i] = vals[i]._replace(rlnImageName=relion_utility.relion_identifier(output, oindex[filename]))
    _logger.info("Stack preprocessing finished")
    _logger.info("Reminder - Using %f angstroms as the diameter of the mask in relion"%(pixel_radius*2*apix))
    return vals

'''
def generate_settings(**extra):

    
    if 'reference' not in extra: extra['reference']=""
    if 'diameter' not in extra: extra['diameter']=extra['pixel_diameter']*extra['apix']
    return """
    Input images: == %(output)s
    Reference map: == %(reference)s
    Particle mask diameter (A): == %(diameter)d
    Pixel size (A): == %(apix)d
    Additional arguments: == --max_memory 32
    """.format(**extra)
'''

"""
def create_refinement(vals, output, **extra):
    '''
    '''
    
    #spider_params.write(os.path.join(os.path.dirname(output), 'params'+os.path.splitext(output)[1]), 0.0, vals[0].rlnVoltage, rlnSphericalAberration, pixel_diameter, window=window, cs=vals[0].rlnAmplitudeContrast)
    align = numpy.zeros((len(vals), 18))
    align[:, 4] = numpy.arange(1, len(vals)+1)
    for i in xrange(len(vals)):
        mic,par = relion_utility.relion_id(vals[i].rlnImageName)
        align[i, 15] = mic
        align[i, 16] = par
        align[i, 17] = vals[i].rlnDefocusU
    format.write(output, align, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus".split(','), format=format.spiderdoc) 
"""

def supports(files, **extra):
    ''' Test if this module is required in the project workflow
    
    :Parameters:
    
    files : list
            List of filenames to test
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    flag : bool
           True if this module should be added to the workflow
    '''
    
    return True

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Relion Selection", "Options to control creation of a relion selection file",  id=__name__)
    group.add_option("-s", class_file="",               help="SPIDER micrograph, class selection file, or comma separated list of classes (e.g. 1,2,3) - if select file does not have proper header, then use `--selection-file filename=id` or `--selection-file filename=id,select`", gui=dict(filetype="open"))

    group.add_option("",   selection_file="",               help="SPIDER micrograph selection file - if select file does not have proper header, then use `--selection-file filename=id` or `--selection-file filename=id,select`", gui=dict(filetype="open"))
    group.add_option("-g", good_file="",                    help="SPIDER particle selection file template - if select file does not have proper header, then use `--good-file filename=id` or `--good-file filename=id,select`", gui=dict(filetype="open"))
    group.add_option("",   reindex_file="",                 help="Reindex file for 1) running movie mode on a subset of selected particles in a stack or 2) When generating a relion selection file from a single stack", gui=dict(filetype="open"))
    group.add_option("-p", param_file="",                   help="SPIDER parameters file (Only required when the input is a stack)", gui=dict(filetype="open"))
    group.add_option("-d", '--ctf-file', defocus_file="",   help="SPIDER defocus file (Only required when the input is a stack)", gui=dict(filetype="open"))
    group.add_option("-l", defocus_header="id:0,defocus:1", help="Column location for micrograph id and defocus value (Only required when the input is a stack)")
    group.add_option("-m", minimum_group=50,                help="Minimum number of particles per defocus group (regroups using the micrograph name)", gui=dict(minimum=0, singleStep=1))
    group.add_option("",   stack_file="",                   help="Used to rename the stack portion of the image name (rlnImageName); ignored when creating a relion file")
    group.add_option("",   scale=1.0,                       help="Used to scale the translations in a relion file")
    group.add_option("",   column="rlnClassNumber",         help="Column name in relion file for selection, e.g. rlnClassNumber to select classes")
    group.add_option("",   test_all=False,                  help="Test the normalization of all the images")
    group.add_option("",   test_valid=False,                help="Test whether the input star file can access all the images")
    group.add_option("",   tilt_pair="",                    help="Selection file that defines pairs of particles (e.g. tilt pairs micrograph1, id1, micrograph2, id2) - outputs a tilted/untilted star files")
    group.add_option("",   min_defocus=4000,                help="Minimum allowed defocus")
    group.add_option("",   max_defocus=100000,              help="Maximum allowed defocus")
    group.add_option("",   random_subset=0,                 help="Split a relion selection file into specificed number of random subsets (0 disables)")
    group.add_option("",   frame_stack_file="",             help="Frame stack filename used to build new relion star file for movie mode refinement. It must have a star for the frame number (e.g. --frame-stack_file frame_*_win_00001.dat)", gui=dict(filetype="open"))
    group.add_option("",   frame_limit=0,                   help="Limit number of frames to use (0 means no limit)")
    group.add_option("",   view_resolution=0,               help="Select a subset to ensure roughly even view distribution (0, default, disables this feature)")
    group.add_option("",   view_limit=0,                    help="Maximum number of projections per view (if 0, then use median)")
    group.add_option("",   downsample=1.0,                  help="Downsample the windows - create new selection file pointing to decimate stacks")
    group.add_option("",   restart=False,                   help="Outputs minimum Relion file to restart refinement from the beginning")
    group.add_option("",   renormalize=0.0,                 help="Diameter of mask in angstroms - overwrites old images with renormalized images")
    group.add_option("",   apix=1.0,                        help="Pixel size of the particle")
    group.add_option("",   invert=False,                    help="Invert the images")
    group.add_option("",   dry_run=False,                   help="Do not process images - only output star file")
    group.add_option("",   single_stack=False,              help="Generate a single stack of images and new relion selection file from an existing relion selection file")
    group.add_option("",   sort_column="",                  help="Sort by column in relion selection file in ascending order")
    group.add_option("",   sort_rev=False,                  help="Reverse sorting to descending order")
    group.add_option("",   split=False,                     help="Split the relion star file using the rlnRandomSubset column")
    group.add_option("",   phase_flip=False,                help="Create a set of phase flipped stacks")
    group.add_option("",   remove_missing=False,            help="Test if image file exists and if not, remove from star file")
    group.add_option("",   mask_diameter=0.0,               help="Mask diameter for Relion (in Angstroms) - 0 mean uses particle_diameter from SPIDER params file")
    group.add_option("",   pad=3,                           help="Padding for image downsize")
    group.add_option("",   image_file="",                   help="Image filename template", gui=dict(filetype="open"))
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", "--particle-file", input_files=[], help="List of filenames for the input images stacks (e.g. cluster/win/win_*.dat) or ReLion selection file (input.star)", required_file=True, gui=dict(filetype="open"), regexp=spider_utility.spider_searchpath)
        pgroup.add_option("-o", "--align-file", output="",      help="Output filename for the relion selection file (Only required if input is a stack)", gui=dict(filetype="save"), required_file=False)
        parser.change_default(log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    
    if ndimage_file.is_readable(options.input_files[0]): #and not format.is_readable(options.input_files[0]):
        if options.defocus_file == "": raise OptionValueError, "No defocus file specified"
        if options.param_file == "": raise OptionValueError, "No parameter file specified"
    
    
    if options.downsample != 1.0 and options.param_file == "": raise ValueError, "Requires SPIDER params file to normalize when using --downsample parameter other than 1.0, --param-file"
    #elif main_option:
    #    if len(options.input_files) != 1: raise OptionValueError, "Only a single input file is supported"

def flags():
    ''' Get flags the define the supported features
    
    :Returns:
    
    flags : dict
            Supported features
    '''
    
    return dict(description = '''Generate a relion selection file from a set of stacks and a defocus file
                         
                         Example: Generate a relion selection file from a set of stacks, defocus file and params file
                         
                         $ ara-selrelion win*.spi -d def_avg.spi -p params.spi -o relion_select.star
                         
                         Example: Select projects in a relion selection file based on the class column using a class selection file
                         
                         $ ara-selrelion relion_select.star -s good_classes.spi relion_select_good.star
                      ''',
                supports_MPI=False, 
                supports_OMP=False,
                use_version=False)

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__)
def dependents(): return []
if __name__ == "__main__": main()


