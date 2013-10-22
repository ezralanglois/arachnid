''' Mask a volume

This |spi| batch file (`spi-mask`) generates a masked volume. It supports:
    
    #. Smoothed spherical masks
    #. Adaptive tight masks
    #. Existing masks in a file

Tips
====

 #. A good value for the density threshold can be found manually in Chimera
 #. The size of a spherical mask is the `pixel-diameter` from the SPIDER params file
 #. The `mask-edge-width` also increases the size of the mask by the given number of pixels
 #. Setting `--volume-mask` to N in this script is an error
 

Examples
========

.. sourcecode :: sh
    
    # Adaptively tight mask a volume
    
    $ spi-mask vol01.spi -o masked_vol01.spi --volume-mask A
    
    # Mask a volume with a cosine smoothed spherical mask
    
    $ spi-mask vol01.spi -o masked_vol01.spi --volume-mask C --mask-edge-width 10
    
    # Mask a volume with a mask in a file
    
    $ spi-mask vol01.spi -o masked_vol01.spi --volume-mask mask_file.spi

Critical Options
================

.. program:: spi-mask

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames containing volumes.
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`
    
.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for masked volume with correct number of digits (e.g. masked_0000.spi)

.. option:: volume-mask <('A', 'C', 'G' or FILENAME)>
    
    Set the type of mask: C for cosine and G for Gaussian and N for no mask and A for adaptive tight mask or a filepath for external mask

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Path to SPIDER params file

.. option:: --bin-factor <FLOAT>
    
    Number of times to decimate params file

Spherical Mask Options
=======================

.. option:: --mask-edge-width <INT>

    Set edge with of the mask (for Gaussian this is the half-width)

Adaptive Tight Mask Options
===========================

.. option:: --threshold <STR or FLOAT>

    Threshold for density or 'A' for auto threshold
    
.. option:: --ndilate <INT>

    Number of times to dilate the mask

.. option:: --gk-size <INT>
 
     Size of the real space Gaussian kernel (must be odd!)

.. option:: --gk-sigma <FLOAT>

    Width of the real space Gaussian kernel

.. option:: --pre_filter <FLOAT>

    Resolution to pre-filter the volume before creating a tight mask (if 0, skip)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by |spi| scripts... <spider-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`


.. Created on Aug 12, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.metadata import spider_params, spider_utility, format_utility
from ..core.image import ndimage_utility, ndimage_file
from ..core.spider import spider, spider_file
import filter_volume
import logging, os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, **extra):
    ''' Create a reference from from a given density map
    
    :Parameters:
    
    filename : str
               Input volume file
    output : str
             Output reference file
    extra : dict
            Unused key word arguments
             
    :Returns:
    
    filename : str
               Filename for correct location
    '''
    
    tempfile1 = extra['spi'].replace_ext('tmp1_spi_file')
    filename1 = spider_file.copy_to_spider(filename, tempfile1)
    if 'apix' in extra:
        extra.update(filter_volume.ensure_pixel_size(filename=filename1, **extra))
    if spider_utility.is_spider_filename(filename[0]):
        output = spider_utility.spider_filename(output, filename[0])
    mask_volume(filename1, output, mask_output=format_utility.add_prefix(output, "mask_"), **extra)
    return filename

def mask_volume(filename, outputfile, spi, volume_mask='N', prefix=None, **extra):
    ''' Mask a volume
    
    :Parameters:
    
    filename : str
               Filename of the input volume
    outputfile : str
                 Filename for output masked volume
    spi : spider.Session
          Current SPIDER session
    volume_mask : str, infile
                  Set the type of mask: C for cosine and G for Gaussian and N for no mask and A for adaptive tight mask or a filename for external mask, F for solvent flattening
    prefix : str
             Prefix for the mask output file
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    outputfile : str
                 Filename for masked volume
    '''
    
    if prefix is not None: outputfile = format_utility.add_prefix(outputfile, prefix)
    mask_type = volume_mask
    if mask_type.find(os.sep) != -1: mask_type = os.path.basename(mask_type)
    mask_type = mask_type.upper()
    _logger.debug("Masking(%s): (%s) %s -> %s"%(mask_type, volume_mask, filename, outputfile))
    if mask_type == 'F':
        flatten(spi, spider.nonspi_file(spi, filename, outputfile), spi.replace_ext(outputfile), **extra)
    elif mask_type == 'A':
        tightmask(spi, spider.nonspi_file(spi, filename, outputfile), spi.replace_ext(outputfile), **extra)
    elif mask_type in ('C', 'G'):
        spherical_mask(filename, outputfile, spi, mask_type, **extra)
    elif mask_type == 'N':
        if outputfile != filename: spi.cp(filename, outputfile)
    elif mask_type != "":
        width = spi.fi_h(filename, ('NSAM', ))[0]
        volume_mask = spider.copy_safe(spi, volume_mask, width)
        spi.mu(filename, volume_mask, outputfile=outputfile)
    else: return filename
    return outputfile

def spherical_mask(filename, outputfile, spi, volume_mask, mask_edge_width=10, pixel_diameter=None, **extra):
    ''' Create a masked volume with a spherical mask
    
    :Parameters:
    
    filename : str
               Filename of the input volume
    outputfile : str
                 Filename for output masked volume
    spi : spider.Session
          Current SPIDER session
    volume_mask : str
                  Set the type of mask: C for cosine and G for Gaussian smoothed spherical mask
    mask_edge_width : int
                      Set edge with of the mask (for Gaussian this is the half-width)
    pixel_diameter : int
                     Diameter of the object in pixels
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    outputfile : str
                 Filename for masked volume
    '''
    
    if filename == outputfile: filename = spi.cp(filename)
    if pixel_diameter is None: raise ValueError, "pixel_diameter must be set with SPIDER params files --param-file"
    width = spider.image_size(spi, filename)[0]/2+1
    radius = pixel_diameter/2+mask_edge_width/2 if volume_mask == 'C' else pixel_diameter/2+mask_edge_width
    return spi.ma(filename, radius, (width, width, width), volume_mask, 'C', mask_edge_width, outputfile=outputfile)

def flatten(spi, filename, outputfile, threshold=0.0, apix=None, mask_output=None, **extra):
    ''' Tight mask the input volume and write to outputfile
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    filename : str
               Input volume
    outputfile : str
                 Output tight masked volume
    threshold : str
                Threshold for density or `A` for auto threshold
    apix : float
           Pixel size
    mask_output : str
                  Output filename for the mask
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    outputfile : str
                 Output tight masked volume
    '''
    
    img = ndimage_file.read_image(filename)
    assert(hasattr(img, 'shape'))
    mask, th = ndimage_utility.flatten_solvent(img, threshold)   
    _logger.info("Flatten solvent to %f"%th) 
    ndimage_file.write_image(outputfile, img*mask)
    return outputfile

def tightmask(spi, filename, outputfile, threshold='A', ndilate=1, gk_size=3, gk_sigma=3.0, pre_filter=0.0, apix=None, mask_output=None, **extra):
    ''' Tight mask the input volume and write to outputfile
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    filename : str
               Input volume
    outputfile : str
                 Output tight masked volume
    threshold : str
                Threshold for density or `A` for auto threshold
    ndilate : int
              Number of times to dilate the mask
    gk_size : int
              Size of the real space Gaussian kernel (must be odd!)
    gk_sigma : float
               Width of the real space Gaussian kernel
    pre_filter : float
                 Resolution to pre-filter the volume before creating a tight mask (if 0, skip)
    apix : float
           Pixel size
    mask_output : str
                  Output filename for the mask
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    outputfile : str
       
                 Output tight masked volume
    '''
    pref_filename=filename
    if pre_filter > 0.0:
        if apix is None and pre_filter > 0.5: raise ValueError, "Filtering requires SPIDER params file --param-file"
        if apix is None and pre_filter < 0.5:
            pref_filename = spi.fq(filename, spi.GAUS_LP, filter_radius=pre_filter, outputfile=format_utility.add_prefix(mask_output, "prefilt_"))
        else: 
            pref_filename = spi.fq(filename, spi.GAUS_LP, filter_radius=apix/pre_filter, outputfile=format_utility.add_prefix(mask_output, "prefilt_"))
        pref_filename = spi.replace_ext(pref_filename)
    
    img = ndimage_file.read_image(pref_filename)
    
    mask = None
    if mask_output is not None and os.path.exists(spi.replace_ext(mask_output)):
        mask_output = spi.replace_ext(mask_output)
        mask = ndimage_file.read_image(mask_output)
        if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1] or mask.shape[2] != img.shape[2]: mask=None
    
    if mask is None:
        try: threshold=float(threshold)
        except: threshold=None
        mask, th = ndimage_utility.tight_mask(img, threshold, ndilate, gk_size, gk_sigma)
        _logger.info("Adaptive mask threshold = %f"%(th))
        if mask_output:
            mask_output = spi.replace_ext(mask_output)
            ndimage_file.write_image(mask_output, mask)
    else:
        _logger.info("Using pre-generated tight-mask: %s"%(mask_output))
        
    img = ndimage_file.read_image(filename)
    ndimage_file.write_image(outputfile, img*mask)
    return outputfile

def apply_mask(filename, outputfile, maskfile):
    ''' Tight mask the input volume and write to outputfile
    
    :Parameters:
    
    filename : str
               Input volume
    outputfile : str
                 Output tight masked volume
    maskfile : str
               Input file containing the mask
    
    :Returns:
    
    outputfile : str
                 Output tight masked volume
    '''
    
    img = ndimage_file.read_image(filename)
    mask = ndimage_file.read_image(maskfile)
    ndimage_file.write_image(outputfile, img*mask)
    return outputfile

def initialize(files, param):
    # Initialize global parameters for the script
    
    param['spi'] = spider.open_session(files, **param)
    if param['param_file'] != "":
        spider_params.read(param['spi'].replace_ext(param['param_file']), param)

def finalize(files, **extra):
    # Finalize global parameters for the script
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import setup_options_from_doc, OptionGroup
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input filenames containing volumes", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for masked volume with correct number of digits (e.g. masked_0000.spi)", gui=dict(filetype="save"), required_file=True)
        spider_params.setup_options(parser, pgroup, False)
    mgroup = OptionGroup(parser, "Masking", "Option to control masking",  id=__name__, group_order=0)
    setup_options_from_doc(parser, mask_volume, spherical_mask, tightmask, group=mgroup)
    pgroup.add_option_group(mgroup)
    if main_option:
        setup_options_from_doc(parser, spider.open_session, group=pgroup)
        parser.change_default(thread_count=4, log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if main_option:
        #spider_params.check_options(options)
        if options.volume_mask == 'N':
            raise OptionValueError, "Invalid parameter: --volume-mask should not be set to 'N', this means no masking"
        if options.volume_mask == "":
            raise OptionValueError, "Invalid parameter: ---volume-mask is empty"
        if not spider_utility.test_valid_spider_input(options.input_files):
            raise OptionValueError, "Multiple input files must have numeric suffix, e.g. vol0001.spi"

def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Mask a volume
                        
                        http://
                        
                        $ %prog vol1.spi vol2.spi -o masked_vol_0001.spi --volume-mask G
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=False,
        use_version = False,
        max_filename_len = 78,
    )
if __name__ == "__main__": main()

