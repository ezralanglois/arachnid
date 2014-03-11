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
from ..core.app import program
from ..core.util.matplotlib_nogui import pylab
from ..core.image import ndimage_file
from ..core.metadata import format, format_utility, spider_utility, spider_params
from ..core.util import fitting
from ..core.spider import spider, spider_file
import mask_volume
import logging, os, numpy

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
    
    spi = extra['spi']
    tempfile1 = spi.replace_ext('tmp1_spi_file')
    tempfile2 = spi.replace_ext('tmp2_spi_file')
    filename1 = spider_file.copy_to_spider(filename[0], tempfile1)
    filename2 = spider_file.copy_to_spider(filename[1], tempfile2)
    filename = (filename1, filename2)
    if spider_utility.is_spider_filename(filename[0]):
        output = spider_utility.spider_filename(output, filename[0])
    sp, fsc, apix = estimate_resolution(filename[0], filename[1], outputfile=output, **extra)
    res = apix/sp if sp > 0 else 0
    res1 = fitting.fit_linear_interp(fsc, 0.5)
    res2 = fitting.fit_linear_interp(fsc, 0.143)
    res1 = apix/res1 if res1 > 0 else 0
    res2 = apix/res2 if res2 > 0 else 0
    _logger.info(" - Resolution = %f - between %s and %s --- (0.5) = %f | (0.143) = %f"%(res, filename[0], filename[1], res1, res2))
    return filename, fsc, apix

def estimate_resolution(filename1, filename2, spi, outputfile, resolution_mask='N', res_edge_width=3, res_threshold='A', res_ndilate=0, res_gk_size=3, res_gk_sigma=5.0, res_filter=0.0, dpi=None, disable_sigmoid=None, disable_scale=None, **extra):
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
    resolution_mask : infile
                      Spherical mask: Set the type of mask: C for cosine and G for Gaussian and N for no mask and A for adaptive tight mask or a filepath for external mask
    res_edge_width : int
                     Spherical mask: Set edge with of the mask (for Gaussian this is the half-width)
    res_threshold : str
                   Tight mask: Threshold for density or `A` for auto threshold
    res_ndilate : int
                  Tight mask: Number of times to dilate the mask
    res_gk_size : int
                  Tight mask: Size of the real space Gaussian kernel (must be odd!)
    res_gk_sigma : float
                   Tight mask: Width of the real space Gaussian kernel
    res_filter : float
                 Resolution to pre-filter the volume before creating a tight mask (if 0, skip)
    dpi : int
          Dots per inch for output plot
    disable_sigmoid : bool
                      Disable the sigmoid model fitting
    extra : dict 
            Unused keyword arguments
    
    :Returns:
    
    sp : float
         Spatial frequency of the volumes
    
    .. todo:: mask based on full volume or half volume average
    '''
    
    for val in "volume_mask,mask_edge_width,threshold,ndilate,gk_size,gk_sigma,prefix".split(','): 
        if val in extra: del extra[val]
    
    extra.update(ensure_pixel_size(spi, filename1, **extra))
    mask_output = format_utility.add_prefix(outputfile, "mask_")
    #_logger.error("apix=%f"%extra['apix'])
    #_logger.error("bin_factor=%f"%extra['bin_factor'])
    #_logger.error("window=%f"%extra['window'])
    if len(res_threshold.split(',')) == 2: 
        res_threshold1, res_threshold2 = res_threshold.split(',')
    else: 
        res_threshold1, res_threshold2 = res_threshold,res_threshold
    filename1 = mask_volume.mask_volume(filename1, outputfile, spi, resolution_mask, mask_edge_width=res_edge_width, threshold=res_threshold1, ndilate=res_ndilate, gk_size=res_gk_size, gk_sigma=res_gk_sigma, pre_filter=res_filter, prefix='res_mh1_', pixel_diameter=extra['pixel_diameter'], apix=extra['apix'], mask_output=mask_output, window=extra['window'])
    filename2 = mask_volume.mask_volume(filename2, outputfile, spi, resolution_mask, mask_edge_width=res_edge_width, threshold=res_threshold2, ndilate=res_ndilate, gk_size=res_gk_size, gk_sigma=res_gk_sigma, pre_filter=res_filter, prefix='res_mh2_', pixel_diameter=extra['pixel_diameter'], apix=extra['apix'], mask_output=mask_output, window=extra['window'])
    dum,pres,sp = spi.rf_3(filename1, filename2, outputfile=outputfile, **extra)
    _logger.debug("Found resolution at spatial frequency: %f"%sp)
    vals = numpy.asarray(format.read(spi.replace_ext(outputfile), numeric=True, header="id,freq,dph,fsc,fscrit,voxels"))
    write_xml(os.path.splitext(outputfile)[0]+'.xml', vals[:, 1], vals[:, 3])
    if pylab is not None:
        plot_fsc(format_utility.add_prefix(outputfile, "plot_"), vals[:, 1], vals[:, 3], extra['apix'], dpi, disable_sigmoid, 0.5, disable_scale)
    return sp, numpy.vstack((vals[:, 1], vals[:, 3])).T, extra['apix']

def write_xml(output, x, y):
    '''
    '''
    fout = open(output, 'w')
    fout.write('<?xml version="1.0" encoding="UTF-8"?>\n') #header
    fout.write('<fsc title="%s" xaxis="%s" yaxis="%s">\n'%('FSC Plot', 'Normalized Spatial Frequency', 'Fourier Shell Correlation')) #FSC Tag
    
    for i in xrange(len(x)):
        fout.write('\t<coordinate>\n\t\t<x>%f</x>\n\t\t<y>%f</y>\n\t</coordinate>\n'%(x[i], y[i]))
    fout.write('</fsc>') #FSC Tag end
    fout.close()

def ensure_pixel_size(spi, filename, **extra):
    ''' Ensure the proper pixel size
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    filename : str
                Filename of the first input volume
    
    :Returns:
    
    params : dict
             Updated SPIDER params
    '''
    
    if extra.get('apix', 0)==0:
        header = ndimage_file.read_header(filename)
        extra['apix']=header['apix']
        if extra['apix']==0: raise ValueError, "Pixel size not in header, must use SPIDER params file!"
    
    del extra['bin_factor']
    try:
        w = spider.image_size(spi, filename)[0]
    except:
        _logger.error("Cannot read: %s -- %d"%(filename, os.path.exists(spi.replace_ext(filename))))
        raise
    w = int(w)
    params = {}
    if extra['window'] != w:
        bin_factor = extra['window']/float(w)
        #extra['dec_level']=1.0
        params = spider_params.update_params(bin_factor, **extra)
        _logger.warn("Changing pixel size: %f (%f/%f) | %f -> %f (%f)"%(bin_factor, extra['window'], w, extra['apix'], params['apix'], extra['dec_level']))
    return params

def plot_fsc(outputfile, x, y, apix, dpi=72, disable_sigmoid=False, freq_rng=0.5, disable_scale=False):
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
    dpi : int
          Resolution of output plot
    disable_sigmoid : bool
                      Disable the sigmoid model fitting
    freq_rng : float, optional
               Spatial frequency range to plot
    '''
    
    if pylab is None: return 
    pylab.switch_backend('Agg')#cairo.png')
    coeff = None
    if not disable_sigmoid:
        try: coeff = fitting.fit_sigmoid(x, y)
        except: _logger.warn("Failed to fit sigmoid, disabling model fitting")
    pylab.clf()
    if coeff is not None:
        pylab.plot(x, fitting.sigmoid(coeff, x), 'g.')
    if disable_scale:
        y -= y.min()
        y /= y.max()
    markers=['r--', 'b--']
    for i, yp in enumerate([0.5, 0.143]):
        if coeff is not None:
            xp = fitting.sigmoid_inv(coeff, yp)
        else:
            xp = fitting.fit_linear_interp((x,y), yp)
        if (apix/xp) < (2*apix) or not numpy.isfinite(xp):
            xp = fitting.fit_linear_interp(numpy.hstack((x[:, numpy.newaxis], y[:, numpy.newaxis])), yp)
        if numpy.alltrue(numpy.isfinite(xp)):
            pylab.plot((x[0], xp), (yp, yp), markers[i])
            pylab.plot((xp, xp), (0.0, yp), markers[i])
            res = 0 if xp == 0 else apix/xp
            pylab.text(xp+xp*0.1, yp, r'$%.3f,\ %.2f \AA (%.2f-criterion)$'%(xp, res, yp))
    
    pylab.plot(x, y)
    if not disable_scale:
        pylab.axis([0.0,freq_rng, 0.0,1.0])
    else:
        pylab.axis([0.0,freq_rng, numpy.min(y), numpy.max(y)])
    pylab.xlabel('Normalized Spatial Frequency')# ($\AA^{-1}$)
    pylab.ylabel('Fourier Shell Correlation')
    #pylab.title('Fourier Shell Correlation')
    pylab.savefig(os.path.splitext(outputfile)[0]+".png", dpi=dpi)

def plot_cmp_fsc(outputfile, fsc_curves, apix, freq_rng=0.5, disable_scale=False):
    '''Write a resolution image plot to a file comparing multiple FSC curves
    
    :Parameters:
    
    outputfile : str
                 Output filename for FSC plot image
    fsc_curves : tuple
                 Spatial frequency array followed by FSC score
    apix : float
           Pixel size
    freq_rng : float, optional
               Spatial frequency range to plot
    '''
    
    if pylab is None: return 
    pylab.switch_backend('cairo.png')
    res = numpy.zeros((len(fsc_curves), 2))
    apix1 = apix
    for i, fsc in enumerate(fsc_curves):
        if isinstance(fsc, tuple):
            if len(fsc) == 3:
                label, fsc, apix1 = fsc
            else:
                label, fsc = fsc
        else: label = "Iteration %d"%(i+1)
        try:
            coeff = fitting.fit_sigmoid(fsc[:, 0], fsc[:, 1])
        except: pass
        else:
            res1 = fitting.sigmoid_inv(coeff, 0.5)
            res2 = fitting.sigmoid_inv(coeff, 0.143)
            if (apix1/res1) < (2*apix1) or not numpy.isfinite(res1) or not numpy.isfinite(res2):
                res1 = fitting.fit_linear_interp(fsc, 0.5)
                res2 = fitting.fit_linear_interp(fsc, 0.143)
            res[i, :] = (apix1/res1, apix1/res2)
            label += ( " $%.3f (%.3f) \AA$"%(apix1/res1, apix1/res2) )
        pylab.plot(fsc[:, 0], fsc[:, 1], label=label)
    
    lgd=pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size':8})
    #pylab.legend(loc=1)
    # detect drop below some number, stop range there?
    if not disable_scale:
        pylab.axis([0.0,0.5,0.0,1.0])
    pylab.xlabel('Normalized Spatial Frequency')# ($\AA^{-1}$)
    pylab.ylabel('Fourier Shell Correlation')
    #pylab.title('Fourier Shell Correlation')
    pylab.savefig(os.path.splitext(outputfile)[0]+".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    #fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    pylab.clf()
    pylab.plot(numpy.arange(1, len(res)+1), res[:, 0], label="FSC(0.5)")
    pylab.plot(numpy.arange(1, len(res)+1), res[:, 1], label="FSC(0.143)")
    lgd=pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size':8})
    #pylab.axis([0.0,0.5,0.0,1.0])
    pylab.ylabel('Resolution ($\AA$)')
    pylab.xlabel('Refinement Step')
    pylab.savefig(os.path.splitext(outputfile)[0]+"_overall.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    

def initialize(files, param):
    # Initialize global parameters for the script
    
    if param['param_file'] != "" and os.path.splitext(param['param_file'])[1] != "":
        _logger.warn("Using extension from SPIDER params file: %s"%param['param_file'])
        sfiles=[param['param_file']]
    else: sfiles=files
    param['spi'] = spider.open_session(sfiles, **param)
    spider_params.read(param['spi'].replace_ext(param['param_file']), param)    
    param['fsc_curves'] = []
    pfiles = []
    if param['sliding']:
        for i in xrange(1, len(files)):
            pfiles.append((files[i-1], files[i]))
    if param['ova']:
        for f in files[:len(files)-1]:
            pfiles.append((f, files[-1]))
    elif param['group']:
        groups = {}
        for f in files:
            id = spider_utility.spider_id(f)
            if id not in groups: groups[id]=[]
            groups[id].append(f)
            if len(groups[id]) > 2: raise ValueError, "Cannot have more than two volumes with the same spider ID: %s"%",".join(groups[id])
        for id in sorted(groups.keys()):
            if len(groups[id]) < 2: raise ValueError, "Cannot have less than two volumes with the same spider ID: %s"%",".join(groups[id])
            pfiles.append(tuple(groups[id]))
    else:
        for i in xrange(0, len(files), 2):
            pfiles.append((files[i], files[i+1]))
    return pfiles

def reduce_all(filename, fsc_curves, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    filename, fsc, apix = filename
    label = os.path.basename(filename[0])
    if spider_utility.is_spider_filename(label):
        label = spider_utility.spider_id(label, use_int=False)
    fsc_curves.append((label, fsc, apix))
    return filename
    

def finalize(files, output, fsc_curves, apix, **extra):
    # Finalize global parameters for the script
    if len(fsc_curves) > 1:
        plot_cmp_fsc(format_utility.add_prefix(output, "plot_cmp_"), fsc_curves, apix)
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import setup_options_from_doc
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input filenames where consecutive names are half volume pairs, must have even number of files", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the doc file contains the FSC curve with correct number of digits (e.g. fsc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-s", sliding=False,  help="Estimate resolution between each neighbor")
        pgroup.add_option("-g", group=False,    help="Group by SPIDER ID")
        pgroup.add_option("",   dpi=72,         help="Resolution of the output plot in dots per inch (DPI)")
        pgroup.add_option("",   disable_sigmoid=False, help="Disable the sigmoid model fitting")
        pgroup.add_option("",   ova=False,      help="One-versus-all, the last one versus all other listed volumes")
        pgroup.add_option("",   disable_scale=False,      help="Scale y-axis automatically")
        
        spider_params.setup_options(parser, pgroup, True)
    setup_options_from_doc(parser, estimate_resolution, 'rf_3', classes=spider.Session, group=pgroup)
    if main_option:
        setup_options_from_doc(parser, spider.open_session, group=pgroup)
        parser.change_default(thread_count=4, log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    path = spider.determine_spider(options.spider_path)
    if path == "": raise OptionValueError, "Cannot find SPIDER executable in %s, please use --spider-path to specify"%options.spider_path
    if main_option:
        #spider_params.check_options(options)
        if not options.ova:
            if len(options.input_files)%2 == 1 and not options.sliding:
                _logger.debug("Found: %s"%",".join(options.input_files))
                raise OptionValueError, "Requires even number of input files or volume pairs - found %d"%len(options.input_files)
            if not spider_utility.test_valid_spider_input(options.input_files[::2]):
                raise OptionValueError, "Multiple input files must have numeric suffix, e.g. vol0001.spi"

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
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

