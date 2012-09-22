''' Calculate the resolution from half-volumes

This |spi| batch file (`spi-resolution`) calculates the resolution from two half-volumes.

Tips
====

 #. The input should be a pair of files `spi-resolution h1rawvol01 h2rawvol01` or
    multiple pairs.
 #. By default, the calculation is done with an adaptive smooth tight mask


Examples
========

.. sourcecode :: sh
    
    # Calculate the resolution between a pair of volumes using the default adaptive tight mask
    
    $ spi-resolution h1_01.spi h2_01.spi -p params
    2012-08-13 09:23:35,634 INFO Resolution = 12.2
    
    # Calculate the resolution between two pairs of volumes using the default adaptive tight mask
    
    $ spi-resolution h1_01.spi h2_01.spi h1_02.spi h2_02.spi -p params
    2012-08-13 09:23:35,634 INFO Resolution = 12.2
    2012-08-13 09:23:35,634 INFO Resolution = 11.1
    
    # Calculate the resolution between two pairs of volumes using the Cosine smoothed spherical mask
    
    $ spi-resolution h1_01.spi h2_01.spi h1_02.spi h2_02.spi -p params --resolution-mask C
    2012-08-13 09:23:35,634 INFO Resolution = 14.5
    2012-08-13 09:23:35,634 INFO Resolution = 13.7

Critical Options
================

.. program:: spi-resolution

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames where consecutive names are half volume pairs, must have even number of files.
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`
    
.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for the doc file contains the FSC curve with correct number of digits (e.g. fsc_0000.spi)

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Path to SPIDER params file

.. option:: --bin-factor <FLOAT>
    
    Number of times to decimate params file

Mask Options
============

.. option:: --resolution-mask : str

    Set the type of mask: C for cosine and G for Gaussian and N for no mask and A for adaptive tight mask or a filepath for external mask

.. option:: --res-edge-width : int

    Set edge with of the mask (for Gaussian this is the half-width)

.. option:: --res-threshold : str

    Threshold for density or 'A' for auto threshold

.. option:: --res-ndilate : int

    Number of times to dilate the mask

.. option:: --res-gk-size : int

    Size of the real space Gaussian kernel (must be odd!)

.. option:: --res-gk-sigma : float

    Width of the real space Gaussian kernel

Resolution Options
==================

These options are passed to SPIDER's `rf_3` command, the default values are generally fine for 
most experiments.

.. option:: --ring-width <float> 
    
    Shell thickness in reciprocal space sampling units (Default: 0.5)

.. option:: --lower-scale <float> 
    
     Lower range of scale factors by which the second Fourier must be multiplied for the comparison (Default: 0.2)

.. option:: --upper-scale <float> 
    
    Upper range of scale factors by which the second Fourier must be multiplied for the comparison (Default: 2.0)

.. option:: --missing-ang <choice('C' or 'W')> 
    
    'C' if you have a missing cone and 'W' if you have a missing wedge (Default: 'C')

.. option:: --max-tilt <float> 
    
    Angle of maximum tilt angle in degrees (Default: 90.0)

.. option:: --noise-factor <float> 
    
    Factor given here determines the FSCCRIT. Here 3.0 corresponds to the 3 sigma criterion i.e., 3/SQRT(N), 
    where N is number of voxels for a given shell.

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
from ..core.image.ndplot import pylab
from ..core.metadata import format, format_utility, spider_utility, spider_params
from ..core.spider import spider
import mask_volume
import logging, os, numpy, scipy

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
    
    if spider_utility.is_spider_filename(filename[0]):
        output = spider_utility.spider_filename(output, filename[0])
    sp = estimate_resolution(filename[0], filename[1], outputfile=output, **extra)
    res = extra['apix']/sp
    _logger.info("Resolution = %f"%(res))
    return filename

def estimate_resolution(filename1, filename2, spi, outputfile, resolution_mask='A', res_edge_width=3, res_threshold='A', res_ndilate=0, res_gk_size=3, res_gk_sigma=3.0, **extra):
    ''' Estimate the resolution from two half volumes
    
    :Parameters:
    
    filename1 : str
                Filename of the first input volume
    filename2 : str
                Filename of the second input volume
    spi : spider.Session
          Current SPIDER session
    outputfile : str
                 Filename for output resolution file (also masks `res_mh1_$outputfile` and `res_mh2_$outputfile` and resolution image `plot_$outputfile.png`)
    resolution_mask : str
                      Set the type of mask: C for cosine and G for Gaussian and N for no mask and A for adaptive tight mask or a filepath for external mask
    res_edge_width : int
                      Set edge with of the mask (for Gaussian this is the half-width)
    res_threshold : str
                Threshold for density or `A` for auto threshold
    res_ndilate : int
              Number of times to dilate the mask
    res_gk_size : int
              Size of the real space Gaussian kernel (must be odd!)
    res_gk_sigma : float
               Width of the real space Gaussian kernel
    extra : dict 
            Unused keyword arguments
    
    :Returns:
    
    sp : float
         Spatial frequency of the volumes
    '''
    
    for val in "volume_mask,mask_edge_width,threshold,ndilate,gk_size,gk_sigma,prefix".split(','): 
        if val in extra: del extra[val]
    filename1 = mask_volume.mask_volume(filename1, outputfile, spi, resolution_mask, mask_edge_width=res_edge_width, threshold=res_threshold, ndilate=res_ndilate, gk_size=res_gk_size, gk_sigma=res_gk_sigma, prefix='res_mh1_', pixel_diameter=extra['pixel_diameter'])
    filename2 = mask_volume.mask_volume(filename2, outputfile, spi, resolution_mask, mask_edge_width=res_edge_width, threshold=res_threshold, ndilate=res_ndilate, gk_size=res_gk_size, gk_sigma=res_gk_sigma, prefix='res_mh2_', pixel_diameter=extra['pixel_diameter'])
    dum,pres,sp = spi.rf_3(filename1, filename2, outputfile=outputfile, **extra)
    if pylab is not None:
        vals = numpy.asarray(format.read(spi.replace_ext(outputfile), numeric=True, header="id,freq,dph,fsc,fscrit,voxels"))
        plot_fsc(format_utility.add_prefix(outputfile, "plot_"), vals[:, 1], vals[:, 3], extra['apix'])
    return sp

def plot_fsc(outputfile, x, y, apix, freq_rng=0.5):
    '''Write a resolution image plot to a file
    
    :Parameters:
    
    outputfile : str
                 Output filename for FSC plot image
    x : array
        Spatial frequency
    y : array
        FSC score
    apix : float
           Pixel size
    freq_rng : float, optional
               Spatial frequency range to plot
    '''
    
    if pylab is None: return 
    try:
        coeff = fit_sigmoid(x, y)
    except: pass
    else:
        pylab.clf()
        pylab.plot(x, sigmoid(coeff, x), 'g.')
        markers=['r--', 'b--']
        for i, yp in enumerate([0.5, 0.14]):
            xp = sigmoid_inv(coeff, yp)
            if numpy.alltrue(numpy.isfinite(xp)):
                pylab.plot((x[0], xp), (yp, yp), markers[i])
                pylab.plot((xp, xp), (0.0, yp), markers[i])
                res = 0 if xp == 0 else apix/xp
                pylab.text(xp+xp*0.1, yp, r'$%.3f,\ %.2f \AA$'%(xp, res))
    
    pylab.plot(x, y)
    pylab.axis([0.0,freq_rng, 0.0,1.0])
    pylab.xlabel('Normalized Frequency')
    pylab.ylabel('Fourier Shell Correlation')
    pylab.title('Fourier Shell Correlation')
    pylab.savefig(os.path.splitext(outputfile)[0]+".png")

def fit_sigmoid(x, y):
    ''' Use non-linear least squares to fit x and y to a sigmoid-like function
    
    :Parameters:
    
    x : array
        X-values for training
    y : array
        y-values for training
    
    :Returns:
    
    coeff : array
            Array of coefficents that fit x to y
    '''
    
    def errfunc(p,x,y): return y-sigmoid(p,x)
    p0 = [0.5, 0.5, 0.5, 0.5]
    return scipy.optimize.leastsq(errfunc,p0,args=(x, y))[0]

def sigmoid_inv(coeff,y):
    ''' Returns a the related inverse of the sigmoid-like function for the given coeffients and x values
    
    :Parameters:
    
    coeff : array
            Array of coeffients
    y : array
        Array of y-values
    
    :Returns:
    
    x : array
        Inverse of sigmoid like function of y-values
    '''
    
    return numpy.log( coeff[1]/(coeff[2]-y) - 1.0 )/-coeff[0] - coeff[3]

def sigmoid(coeff,x):
    ''' Returns a sigmoid-like function for the given coeffients and x values
    
    :Parameters:
    
    coeff : array
            Array of coeffients
    x : array
        Array of x-values
    
    :Returns:
    
    y : array
        Sigmoid like function of x values
    '''
    
    return coeff[2] - ( coeff[1] / (1.0 + numpy.exp(-coeff[0]*(x+coeff[3])) ))

def initialize(files, param):
    # Initialize global parameters for the script
    
    param['spi'] = spider.open_session(files, **param)
    spider_params.read(param['spi'].replace_ext(param['param_file']), param)
    pfiles = []
    for i in xrange(0, len(files), 2):
        pfiles.append((files[i], files[i+1]))
    return pfiles

def finalize(files, **extra):
    # Finalize global parameters for the script
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import setup_options_from_doc
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input filenames where consecutive names are half volume pairs, must have even number of files", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the doc file contains the FSC curve with correct number of digits (e.g. fsc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        spider_params.setup_options(parser, pgroup, True)
    setup_options_from_doc(parser, estimate_resolution, 'rf_3', classes=spider.Session, group=pgroup)
    if main_option:
        setup_options_from_doc(parser, spider.open_session, group=pgroup)
        parser.change_default(thread_count=4, log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if main_option:
        spider_params.check_options(options)
        if len(options.input_files)%2 == 1:
            _logger.debug("Found: %s"%",".join(options.input_files))
            raise OptionValueError, "Requires even number of input files or volume pairs - found %d"%len(options.input_files)
        if not spider_utility.test_valid_spider_input(options.input_files):
            raise OptionValueError, "Multiple input files must have numeric suffix, e.g. vol0001.spi"

def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Calculate the resolution from volume pairs
                        
                        http://
                        
                        $ %prog h1_vol_01.spi h2_vol_02.spi -o resolution_curve.txt
                        
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

