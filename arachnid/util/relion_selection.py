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
    
    $ ara-selrelion win/win_*.spi --defocus def_avg.spi --param-file params.spi -o relion_selection.star --select good_00000.spi
    
    # Example usage - Generate a ReLion selection file from a set of stacks with manual selection where selection was produced by spider and does not have a header
    
    $ ara-selrelion win/win_*.spi --defocus def_avg.spi --param-file params.spi -o relion_selection.star --select good_00000.spi=id,select
    
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
    
.. option:: -s<str>, --select <str>
    
    SPIDER selection file (Only required when the input is a relion selection file) - if select file does not have proper header, then use `--select filename=id` or `--select filename=id,select`
    
Useful Options
==============
    
.. option:: -l <str>, --defocus-header <str>
    
    Column location for micrograph id and defocus value (Default: id:0,defocus:)
    
    Example defocus file
    
    | ;             id     defocus   astig_ang   astig_mag cutoff_freq
    | 1  5       21792     29654.2     34.2511     409.274     0.20764
    | 2  5       21794     32612.5     41.0366     473.659    0.201649

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    
.. Created on Nov 27, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.metadata import spider_utility, format_utility, format, spider_params
from ..core.image import ndimage_utility, ndimage_file
import numpy, os, logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, **extra):
    '''Assign orientation to a set of noisy projections
    
    :Parameters:
        
        filename : tuple
                   File index and input filename
        extra : dict
                Unused key word arguments
    '''
    
    if ndimage_file.is_readable(files[0]):
        _logger.info("Generating a relion selection file from a set of stacks")
        img = ndimage_file.read_image(files[0])
        #img = ndimage_file.read_image(files[0])
        generate_relion_selection_file(files, img, **extra)
    else:
        try:
            _logger.info("Generating a relion selection file from a set of stacks")
            vals = format.read(files[0], numeric=True)
        except:
            raise ValueError, "Input not an image or a selection file"
        else:
            select_class_subset(vals, **extra)
    _logger.info("Completed")
            
def select_class_subset(vals, select, output, **extra):
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
    
    if select != "":
        select = format.read(select, numeric=True)
        select = set([s.id for s in select if s.select > 0])
        subset=[]
        for v in vals:
            if v.rlnClassNumber in select:
                subset.append(v)
    else: subset = vals
    if os.path.splitext(output)[1] == '.star':
        format.write(output, subset)
    else:
        micselect={}
        for v in subset:
            mic,par = spider_utility.relion_id(v.rlnImageName)
            if mic not in micselect: micselect[mic]=[]
            micselect[mic].append((par, 1))
        for mic,vals in micselect.iteritems():
            format.write(output, numpy.asarray(vals), header="id,select".split(','))

def generate_relion_selection_file(files, img, output, defocus, defocus_header, param_file, select="", **extra):
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
    
    spider_params.read_spider_parameters_to_dict(param_file, extra)
    pixel_radius = int(extra['pixel_diameter']/2.0)
    if img.shape[0]%2 != 0: raise ValueError, "Relion requires even sized images"
    if img.shape[0] != img.shape[0]: raise ValueError, "Relion requires square images"
    if pixel_radius > 0:
        mask = ndimage_utility.model_disk(pixel_radius, img.shape[0])*-1+1
        avg = numpy.mean(img*mask)
        if numpy.allclose(0.0, avg):
            raise ValueError, "Relion requires the background to be zero normalized, not %f"%avg
    
    defocus_dict = format.read(defocus, header=defocus_header, numeric=True)
    defocus_dict = format_utility.map_object_list(defocus_dict)
    spider_params.read_spider_parameters_to_dict(param_file, extra)
    voltage, cs, ampcont=extra['voltage'], extra['cs'], extra['ampcont']
    label = []
    idlen = len(str(ndimage_file.count_images(files)))
    for filename in files:
        mic = spider_utility.spider_id(filename)
        if select != "":
            select_vals = format.read(select, spiderid=mic, numeric=True)
            if len(select_vals) > 0 and not hasattr(select_vals[0], 'id'):
                raise ValueError, "Error with selection file (`--select`) missing `id` in header, return with `--select filename=id` or `--select filename=id,select` indicating which column has the id and select"
            if len(select_vals) > 0 and 'select' in select_vals[0]._fields:
                select_vals = [s.id for s in select_vals if s.select > 0] if len(select_vals) > 0 and hasattr(select_vals[0], 'select') else [s.id for s in select_vals]
        else:
            select_vals = xrange(ndimage_file.count_images(filename))
        for pid in select_vals:
            label.append( ("%s@%s"%(str(pid).zfill(idlen), filename), filename, defocus_dict[mic].defocus, voltage, cs, ampcont) )
    format.write(output, label, header="rlnImageName,rlnMicrographName,rlnDefocusU,rlnVoltage,rlnSphericalAberration,rlnAmplitudeContrast")

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Relion Selection", "Options to control creation of a relion selection file",  id=__name__) if pgroup is None else pgroup
    group.add_option("-s", select="",                       help="SPIDER selection file (Only required when the input is a relion selection file) - if select file does not have proper header, then use `--select filename=id` or `--select filename=id,select`")
    group.add_option("-p", param_file="",                   help="SPIDER parameters file (Only required when the input is a stack)")
    group.add_option("-d", defocus="",                      help="SPIDER defocus file (Only required when the input is a stack)")
    group.add_option("-l", defocus_header="id:0,defocus:1", help="Column location for micrograph id and defocus value (Only required when the input is a stack)")
    parser.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input stacks or selection file", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the relion selection file", gui=dict(filetype="save"), required_file=True)
        pgroup.change_default(log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    
    if ndimage_file.is_readable(options.input_files[0]):
        if options.defocus == "": raise OptionValueError, "No defocus file specified"
        if options.param_file == "": raise OptionValueError, "No parameter file specified"
    if main_option:
        if len(options.input_files) != 1: raise OptionValueError, "Only a single input file is supported"

def main():
    #Main entry point for this script
    from ..core.app.program import run_hybrid_program
    
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


