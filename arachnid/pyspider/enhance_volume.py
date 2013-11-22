''' Volume filtering and amplitude enhancement

This pySPIDER batch file (`spi-enhance`) filters (and enhances) a volume to a specific resolution.

Tips
====

 #. Boolean flags such as `disable-enhance` cannot take a value on the command-line. They only require a value in a configuration 
    file. Their existence or absence sets the appropriate value. For example, specifiying 
    `$ spi-filtervol --disable-enhance ....` will disable amplitude enhancement.

 #. This script interpolates passed the last value in the scatter file. It uses a linear model that starts at 14 A and goes to 
    the last available resolution.
    
Examples
========

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Filter and enhance several volumes
    
    $ spi-enhance vol1.spi vol2.spi -p params.spi -o filt_vol_0001.spi -r 15 --scatter-doc scattering8.spi

Critical Options
================

.. program:: spi-enhance

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames containing volumes.
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`
    
.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for filtered/enhanced volume with correct number of digits (e.g. masked_0000.spi)
    
.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Path to SPIDER params file

.. option:: --bin-factor <FLOAT>
    
    Number of times to decimate params file

.. option:: --resolution <FLOAT>
    
    Resolution to filter the volumes

.. option:: --scatter-doc <FILENAME>
    
    Filename for x-ray scatter file

Tight Masking Options
=====================

.. program:: spi-enhance

.. option:: --enh-mask <BOOL>

    Generate a tight mask under which to calculate the fall off

.. option:: --enh-filter <FLOAT>

    Gaussian lowpass filter to given resolution for tight mask (0 disables)

.. option:: --enh-gk-sigma <FLOAT>

    Gaussian filter width in real space for tight mask (0 disables)

.. option:: --enh-gk-size <INT>
    
    Size of the real space Gaussian kernel for tight mask (must be odd!)

.. option:: --enh-threshold <'A' or FLOAT>

    Threshold for density or `A` for auto threshold for tight mask

.. option:: --enh-ndilate <INT>

    Number of times to dilate the mask for tight mask

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
from ..core.app import program
from ..core.metadata import spider_params, spider_utility, format_utility, format
from ..core.spider import spider
import mask_volume, filter_volume
import logging, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, spi, output, resolution, **extra):
    ''' Create a reference from from a given density map
    
    :Parameters:
    
    filename : str
               Input volume file
    output : str
             Output reference file
    resolution : float
                 Resolution to filter
    extra : dict
            Unused key word arguments
             
    :Returns:
    
    filename : str
               Filename for correct location
    '''
    
    # test if enh exists!
    extra.update(filter_volume.ensure_pixel_size(spi, filename, **extra))
    if spider_utility.is_spider_filename(filename):
        output = spider_utility.spider_filename(output, filename)
    if resolution > 0:
        sp = extra['apix']/resolution
        filename = filter_volume.filter_volume_lowpass(filename, spi, sp, outputfile=output, **extra)
    output = enhance_volume(filename, spi, sp, output, **extra)
    return filename

def enhance_volume(filename, spi, sp, outputfile, scatter_doc="", enh_mask=False, enh_gk_sigma=9.0, enh_gk_size=3, enh_filter=0.0, enh_threshold='A', enh_ndilate=1, apix=None, window=None, prefix=None, **extra):
    '''Frequency enhance volume
    
    :Parameters:

    filename : str
               Filename of the input volume
    spi : vispider.core.spider.Session
              Current SPIDER session
    sp : float
         Spatial frequency to filter volume
    outputfile : str
                 Filename of the output volume
    scatter_doc : str
                  Filename for x-ray scatter file
    enh_mask : bool
               Generate a tight mask under which to calculate the fall off
    enh_gk_sigma : float
                   Gaussian filter width in real space for tight mask (0 disables)
    enh_gk_size : int
                  Size of the real space Gaussian kernel for tight mask (must be odd!)
    enh_filter : float
                 Resolution to pre-filter the volume before creating a tight mask (if 0, skip)
    enh_threshold : str
                    Threshold for density or `A` for auto threshold  for tight mask
    enh_ndilate : int
                  Number of times to dilate the mask  for tight mask
    apix : float
           Pixel size (provided by the SPIDER params file)
    window : int
             Size of the current window (provided by the SPIDER params file)
    prefix : str
             Prefix for the mask output file
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    output_volume : str
                    Name of the output volume
    '''
    if scatter_doc == "": return filename
    if prefix is not None: outputfile = format_utility.add_prefix(outputfile, prefix)
    if filename == outputfile: filename = spi.cp(filename)
    
    window, = spi.fi_h(filename, ('NSAM'))
    filter_limit = int(window*sp) # Thanks to Jesper Pallesen
    _logger.info("Enhancing with filter limit: %d"%(filter_limit))
    tmp_roo = format_utility.add_prefix(outputfile, 'roo')
    spi.de(tmp_roo)
    
    if enh_mask:
        temp = filter_volume.filter_volume_lowpass(spi, filename, apix/enh_filter, 3) if enh_filter > 0 else filename
        mvol = format_utility.add_prefix(outputfile, 'enh_tm_')
        mask_volume.tightmask(spi, spider.nonspi_file(spi, temp, mvol), spi.replace_ext(mvol), enh_threshold, enh_ndilate, enh_gk_size, enh_gk_sigma, enh_filter, apix)
        mvol = spi.cp(mvol)
        spi.de(mvol)
    else:
        mvol = filename
    
    pvol = spi.pw(mvol)
    if enh_mask: spi.de(mvol)
    spvol = spi.sq(pvol)
    rot_avg = spi.ro(spvol)
    spi.li_d(rot_avg, 'R', 1, outputfile=tmp_roo, use_2d=False)
    powspec = format.read(spi.replace_ext(tmp_roo), ndarray=True)[0]
    scatter = format.read(spi.replace_ext(scatter_doc), ndarray=True)[0]
    scatter = scatter[:, 1:]
    powspec = powspec[:, 1:]
    outvals = numpy.zeros((filter_limit, 4))
    
    idx = len(scatter)-numpy.searchsorted(scatter[::-1, 0], 14)-1
    for i in xrange(1, filter_limit+1):
        res = float(i)/(2.0*(len(powspec)-1))
        sfreq = apix/res
        idx = numpy.searchsorted(scatter[::-1, 0], sfreq)
        cur_col2 = scatter[len(scatter)-idx-1, 1] 
        if sfreq < scatter[len(scatter)-1, 0]:
            _logger.warn("Exceeded 7.85: %f -> %d: %f"%(sfreq, len(scatter)-idx-1, cur_col2))
        cur_col2 /= 37891.0
        cur_col2 = numpy.sqrt(cur_col2/powspec[i, 0])
        outvals[i-1, :] = (cur_col2, res, sfreq, numpy.log(cur_col2))
    
    tmp_roo = format_utility.add_prefix(outputfile, 'scatter')
    format.write(spi.replace_ext(tmp_roo), outvals, header="c1,c2,c3,c4".split(','), format=format.spiderdoc)
    return spi.fd(filename, tmp_roo, outputfile)

def initialize(files, param):
    # Initialize global parameters for the script
    
    param['spi'] = spider.open_session(files, **param)
    spider_params.read(param['spi'].replace_ext(param['param_file']), param)
    
    _logger.info("Pixel size: %f"%param['apix'])
    _logger.info("Bin factor: %f"%param['bin_factor'])

def finalize(files, **extra):
    # Finalize global parameters for the script
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import setup_options_from_doc
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input filenames containing volumes", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for filtered/enhanced volume with correct number of digits (e.g. masked_0000.spi)", gui=dict(filetype="save"), required_file=True)
        spider_params.setup_options(parser, pgroup, True)
        pgroup.add_option("-r", resolution=15.0,     help="Resolution to filter the volumes")
        setup_options_from_doc(parser, spider.open_session, group=pgroup)
    
    
    setup_options_from_doc(parser, enhance_volume, group=pgroup)
    parser.change_default(thread_count=4, log_level=3)
    

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if main_option:
        spider_params.check_options(options)
        if options.scatter_doc == "": raise OptionValueError, "Missing argument, no scattering file specified - use `--scatter-doc <filename>` "
        if not spider_utility.test_valid_spider_input(options.input_files):
            raise OptionValueError, "Multiple input files must have numeric suffix, e.g. vol0001.spi"

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Filter and enhance a volume
                        
                        http://
                        
                        $ %prog vol1.spi vol2.spi -p params.spi -o vol_0001.spi -r 15 --scatter-doc scattering8
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=False,
        use_version = False,
        max_filename_len = 78,
    )
def dependents(): return [filter_volume]
if __name__ == "__main__": main()





