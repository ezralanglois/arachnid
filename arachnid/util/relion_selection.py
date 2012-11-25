''' Generate selection files to interface SPIDER and RELION

This script (`ara-selrelion`) can generate a RELION selection file from a list of stacks, a defocus
file and a SPIDER params file. It can also take a RELION selection file and a SPIDER selection file
and generate a new RELION selection file. Finally, it can generate a set of SPIDER selection files (by micrograph)
from a RELION selection file.

Tips
====

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
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Example usage - Generate a ReLion selection file from a set of stacks
    
    $ ara-selrelion win/win_*.spi --defocus def_avg.spi --param-file params.spi -o relion_selection.star
    
    # Example usage - Generate a ReLion selection file from a set of stacks with manual selection
    
    $ ara-selrelion win/win_*.spi --defocus def_avg.spi --param-file params.spi -o relion_selection.star --good good_00000.spi
    
    # Example usage - Generate a ReLion selection file from a set of stacks with manual selection where selection was produced by spider and does not have a header
    
    $ ara-selrelion win/win_*.spi --defocus def_avg.spi --param-file params.spi -o relion_selection.star --good good_00000.spi=id,select
    
    # Example usage - Generate a ReLion selection file from a SPIDER selection file and a Relion selection file
    
    $ ara-selrelion relion_selection_full.star --select good_classavg01.spi -o relion_selection_subset.star
    
    # Example usage - Generate SPIDER selection files by micrograph from a Relion and SPIDER selection file
    
    $ ara-selrelion relion_selection_full.star --select good_classavg01.spi -o good_000001.spi


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

.. option:: -p <str>, --param-file  <str>
    
    SPIDER parameters file (Only required when the input is a stack)
    
.. option:: -d <str>, --defocus <str>
    
    SPIDER defocus file (Only required when the input is a stack)
    
.. option:: -s <str>, --select <str>
    
    SPIDER micrograph or class selection file - if select file does not have proper header, then use `--select filename=id` or `--select filename=id,select`
    
.. option:: -g <str>, --good <str>
    
    SPIDER particle selection file (used when creating a new relion selection file) - if select file does not have proper header, then use `--select filename=id` or `--select filename=id,select`
    
Useful Options
==============
    
.. option:: -l <str>, --defocus-header <str>
    
    Column location for micrograph id and defocus value (Default: id:0,defocus:)
    
    Example defocus file
    
    | ;             id     defocus   astig_ang   astig_mag cutoff_freq
    | 1  5       21792     29654.2     34.2511     409.274     0.20764
    | 2  5       21794     32612.5     41.0366     473.659    0.201649

.. option:: -m <INT>, --minimum-group <INT>
    
    Minimum number of particles per defocus group

.. option:: --stack_file <STR>

    Used to rename the stack portion of the image name (rlnImageName); ignored when creating a relion file

.. option:: --scale <FLOAT>
    
    Used to scale the translations in a relion file (Default: 1.0)

.. option:: --column <str>
    
    Column name in relion file for selection, e.g. rlnClassNumber to select classes (Default: rlnClassNumber)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    
.. Created on Nov 27, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.metadata import spider_utility, format_utility, format, spider_params
from ..core.image import eman2_utility, ndimage_file #, ndimage_utility
import numpy, os, logging, operator

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, **extra):
    '''Generate a relion selection file
    
    :Parameters:
    
    filename : tuple
               File index and input filename
    extra : dict
            Unused key word arguments
    '''
    
    if not format.is_readable(files[0]) and ndimage_file.is_readable(files[0]):
        _logger.info("Generating a relion selection file from a set of stacks")
        img = ndimage_file.read_image(files[0])
        #img = ndimage_file.read_image(files[0])
        generate_relion_selection_file(files, img, **extra)
    else:
        _logger.info("Transforming a relion selection file: %d"%len(files))
        vals = []
        for f in files:
            try:
                vals.append(format.read(f, numeric=True))
            except:
                raise ValueError, "Input not an image or a selection file"
        if len(vals) > 1:
            vals = format_utility.combine(vals)
        else: vals = vals[0]
        

        select_class_subset(vals, **extra)
    _logger.info("Completed")
            
def select_class_subset(vals, select, output, column="rlnClassNumber", **extra):
    ''' Select a subset of classes and write a new selection file
    
    :Parameter:
    
    vals : list
           List of entries from a selection file
    select : str
             Filename for good class selection file
    output : str
            Filename for output selection file
    extra : dict
            Unused key word arguments
    '''
    
    if len(vals) == 0: raise ValueError, "No values read"
    if select != "":
        select = format.read(select, numeric=True)
        select = set([s.id for s in select if s.select > 0])
        subset=[]
        for v in vals:
            id = getattr(v, column)
            try: id = int(id)
            except: id = spider_utility.spider_id(id)
            if id in select: subset.append(v)
        if len(subset) == 0: raise ValueError, "No classes selected"
    else: subset = vals
    if os.path.splitext(output)[1] == '.star':       
        groupmap = regroup(build_group(subset), **extra)
        update_parameters(subset, list(subset[0]._fields), groupmap, **extra)
        format.write(output, subset)
    else:
        micselect={}
        for v in subset:
            mic,par = spider_utility.relion_id(v.rlnImageName)
            if mic not in micselect: micselect[mic]=[]
            micselect[mic].append((par, 1))
        for mic,vals in micselect.iteritems():
            format.write(output, numpy.asarray(vals), spiderid=mic, header="id,select".split(','), format=format.spidersel)

def generate_relion_selection_file(files, img, output, defocus, defocus_header, param_file, select="", good="", **extra):
    ''' Generate a relion selection file for a list of stacks, defocus file and params file
    
    :Parameters:

    files : list
            List of stack files
    img : EMData
          Image used to query size information
    output : str
             Filename for output selection file
    defocus : str
              Filename for input defocus file
    defocus_header : str
                     Header for defocus file
    param_file : str
                 Filename for input SPIDER Params file
    select : str
             Filename for input optional selection file (for good particles in each stack)
    extra : dict
            Unused key word arguments
    '''
    
    header = "rlnImageName,rlnMicrographName,rlnDefocusU,rlnVoltage,rlnSphericalAberration,rlnAmplitudeContrast,rlnGroupNumber".split(',')
    spider_params.read(param_file, extra)
    pixel_radius = int(extra['pixel_diameter']/2.0)
    if img.shape[0]%2 != 0: raise ValueError, "Relion requires even sized images"
    if img.shape[0] != img.shape[0]: raise ValueError, "Relion requires square images"
    if pixel_radius > 0:
        #mask = ndimage_utility.model_disk(pixel_radius, img.shape[0])*-1+1
        mask = eman2_utility.model_circle(pixel_radius, img.shape[0], img.shape[1])*-1+1
        avg = numpy.mean(img*mask)
        if numpy.allclose(0.0, avg):
            _logger.warn("Relion requires the background to be zero normalized, not %g"%avg)
    
    defocus_dict = format.read(defocus, header=defocus_header, numeric=True)
    defocus_dict = format_utility.map_object_list(defocus_dict)
    if select != "": 
        select = format.read(select, numeric=True)
        old = defocus_dict
        defocus_dict = {}
        for s in select: defocus_dict[s.id]=old[s.id]
    spider_params.read(param_file, extra)
    voltage, cs, ampcont=extra['voltage'], extra['cs'], extra['ampcont']
    label = []
    idlen = len(str(ndimage_file.count_images(files)))
    group = []
    for filename in files:
        mic = spider_utility.spider_id(filename)
        if good != "":
            if not os.path.exists(spider_utility.spider_filename(good, mic)): continue
            select_vals = format.read(good, spiderid=mic, numeric=True)
            if len(select_vals) > 0 and not hasattr(select_vals[0], 'id'):
                raise ValueError, "Error with selection file (`--select`) missing `id` in header, return with `--select filename=id` or `--select filename=id,select` indicating which column has the id and select"
            if len(select_vals) > 0 and 'select' in select_vals[0]._fields:
                select_vals = [s.id for s in select_vals if s.select > 0] if len(select_vals) > 0 and hasattr(select_vals[0], 'select') else [s.id for s in select_vals]
        else:
            select_vals = xrange(1, ndimage_file.count_images(filename)+1)
        if mic not in defocus_dict:
            _logger.warn("Micrograph not found in defocus file: %d -- skipping"%mic)
            continue
        if defocus_dict[mic].defocus < 1000: 
            _logger.warn("Micrograph %d defocus too small: %f -- skipping"%(mic, defocus_dict[mic].defocus))
            continue
        
        group.append((defocus_dict[mic].defocus, len(select_vals), len(label), mic))
        for pid in select_vals:
            label.append( ["%s@%s"%(str(pid).zfill(idlen), filename), filename, defocus_dict[mic].defocus, voltage, cs, ampcont, len(group)-1] )
    if len(group) == 0: raise ValueError, "No values to write out, try changing selection file"
    groupmap = regroup(group, **extra)
    update_parameters(label, header, groupmap, **extra)
    
    format.write(output, label, header=header)
    
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
            groupmap[int(group[i, 3])]=offset+1
            regroup.append(i)
            total += group[i, 1]
        if total > minimum_group:
            offset += 1
            total = 0
            regroup=[]
    if total < minimum_group:
        for i in regroup:
            groupmap[i]=offset
    _logger.info("Groups %d -> %d"%(len(group), offset))
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
    if scale != 1.0:
        try:x_col = header.index('rlnOriginX')
        except: x_col = -1
        try:y_col = header.index('rlnOriginY')
        except: y_col = -1
    else: x_col, y_col = -1, -1
    name_col = header.index('rlnImageName') if stack_file != "" else -1
    Tuple = data[0].__class__
    for i in xrange(len(data)):
        vals = data[i] if not isinstance(data[i], tuple) else list(data[i])
        if group_col >= 0 and group_map is not None:
            id = spider_utility.spider_id(vals[group_col])
            try:
                vals[group_col] = spider_utility.spider_filename(vals[group_col], group_map[id])
            except:
                _logger.error("keys: %s"%str(group_map.keys()))
                raise
        if x_col > -1: vals[x_col]*= scale
        if y_col > -1: vals[y_col]*= scale
        if name_col > -1: vals[name_col] = spider_utility.relion_filename(stack_file, vals[name_col])
        if hasattr(Tuple, '_make'):
            data[i] = Tuple._make(vals)
        else: data[i] = tuple(vals)
    return data

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Relion Selection", "Options to control creation of a relion selection file",  id=__name__)
    group.add_option("-s", select="",                       help="SPIDER micrograph or class selection file - if select file does not have proper header, then use `--select filename=id` or `--select filename=id,select`", gui=dict(filetype="open"))
    group.add_option("-g", good="",                         help="SPIDER particle selection file (used when creating a new relion selection file) - if select file does not have proper header, then use `--select filename=id` or `--select filename=id,select`", gui=dict(filetype="open"))
    group.add_option("-p", param_file="",                   help="SPIDER parameters file (Only required when the input is a stack)", gui=dict(filetype="open"))
    group.add_option("-d", defocus="",                      help="SPIDER defocus file (Only required when the input is a stack)", gui=dict(filetype="open"))
    group.add_option("-l", defocus_header="id:0,defocus:1", help="Column location for micrograph id and defocus value (Only required when the input is a stack)")
    group.add_option("-m", minimum_group=20,                help="Minimum number of particles per defocus group", gui=dict(minimum=0, singleStep=1))
    group.add_option("",   stack_file="",                   help="Used to rename the stack portion of the image name (rlnImageName); ignored when creating a relion file")
    group.add_option("",   scale=1.0,                       help="Used to scale the translations in a relion file")
    group.add_option("",   column="rlnClassNumber",         help="Column name in relion file for selection, e.g. rlnClassNumber to select classes")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input stacks or selection file", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the relion selection file", gui=dict(filetype="save"), required_file=True)
        parser.change_default(log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    
    if not format.is_readable(options.input_files[0]) and ndimage_file.is_readable(options.input_files[0]):
        if options.defocus == "": raise OptionValueError, "No defocus file specified"
        if options.param_file == "": raise OptionValueError, "No parameter file specified"
    #elif main_option:
    #    if len(options.input_files) != 1: raise OptionValueError, "Only a single input file is supported"

def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Generate a relion selection file from a set of stacks and a defocus file
        
                         http://
                         
                         Example: Generate a relion selection file from a set of stacks, defocus file and params file
                         
                         $ ara-selrelion win*.spi -d def_avg.spi -p params.spi -o relion_select.star
                         
                         Example: Select projects in a relion selection file based on the class column using a class selection file
                         
                         $ ara-selrelion relion_select.star -s good_classes.spi relion_select_good.star
                      ''',
        supports_MPI = False,
        use_version = False,
    )
def dependents(): return []
if __name__ == "__main__": main()


