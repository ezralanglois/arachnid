''' Estimate the defocus of a micrograph or a stack of particles

This |spi| batch file (`spi-defocus`) estimates the defocus of a set of micrographs or particle stacks.

Tips
====

 #. The micrographs do not have to be in SPIDER format (automatic conversion is attempted). However, 
    :option:`--data-ext` must be used to set the appropriate SPIDER extension.
 
 #. The `bin-factor` options controls the size of multiple parameters, see documentation below for more
    information.
    
Examples
========

.. sourcecode :: sh
    
    # Estimate the defocus over a set of micrographs
    
    $ spi-defocus mic_*.ter -p params.ter -o defocus.ter
    
    # Estimate the defocus over a set of micrographs in TIFF format
    
    $ spi-defocus mic_*.tif -p params.ter -o defocus.ter --data-ext ter
    
    # Esimtate the defocus over a set of image stacks
    
    $ spi-defocus stack_*.ter -p params.ter -o defocus.ter
    
    # Esimtate the defocus over a set of power spectra
    
    $ spi-defocus pow_*.ter -p params.ter -o defocus.ter --use-powerspec

Critical Options
================

.. program:: spi-defocus

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames containing micrographs, window stacks or power spectra.
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`

.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for defocus file with correct number of digits (e.g. sndc_0000.spi)
    
.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Path to SPIDER params file

.. option:: --bin-factor <float>
    
    Number of times to decimate params file, and parameters: `window-size`, `x-dist, and `x-dist` and optionally the micrograph

Useful Options
==============

.. option:: --window-size <INT> 
    
    Size of the window to be cropped from the micrograph for the power spectra (Default: 500)

.. option:: --pad <INT> 
    
    Number of times to pad window before Fourier transform; if pad > window-size, then size of padded image (0 means none)

.. option:: --disable-bin <BOOL>
    
    Disable micrograph decimation

Periodogram Options
===================

.. option:: --x-overlap <INT> 
    
    Percent overlap in the x-direction (Default: 50)

.. option:: --y-overlap <INT> 
    
    Percent overlap in the y-direction (Default: 50)

.. option:: --x-dist <INT> 
    
    Distance from the edge in the x-direction (Default: 0)

.. option:: --y-dist <INT> 
    
    Distance from the edge in the y-direction (Default: 0)

Advanced Options
================

.. option:: --output-pow <FILENAME> 
    
    Filename for output power spectra (Default: pow/pow_00000)

.. option:: --output-roo <FILENAME> 
    
    Filename for output rotational average (Default: roo/roo_00000)

.. option:: --output-ctf <FILENAME> 
    
    Filename for output CTF curve (Default: ctf/ctf_00000)

.. option:: --use-powerspec <BOOL>
    
    Set True if the input file is a power spectra (Default: False)

.. option:: --du-nstd <LIST>
    
    List of number of standard deviations for dedusting (Default: empty)

.. option:: --du-type <CHOICE>
    
    Type of dedusting: (1) BOTTOM, (2) TOP, (3) BOTH SIDES: 3 (Default: 3)

.. option:: --ps-radius <INT>
    
    Power spectrum mask radius (Default: 225)

.. option:: --ps-outer <INT>
    
    Power spectrum outer mask radius (Default: 0)

.. option:: --inner-radius <INT> 
    
    Inner mask size for power spectra enhancement (Default: 5)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by MPI-enabled scripts... <mpi-options>`
    #. :ref:`Options shared by |spi| scripts... <spider-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`

.. todo:: create fast version that does all the power spect averaging in numpy

.. Created on Jul 15, 2011
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.metadata import format, spider_params, spider_utility, format_utility
from ..core.image import ndimage_file, eman2_utility, ndimage_utility, analysis
from ..core.parallel import mpi_utility
from ..core.spider import spider
from ..core.util import plotting
import os, numpy, logging, itertools #, scipy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, id_len=0, **extra):
    ''' Esimate the defocus of the given micrograph
    
    :Parameters:
    
    filename : str
               Input micrograph, stack or power spectra file
    output : str
             Output defocus file
    id_len : int
             Max length of SPIDER ID
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    filename : str
               Current filename
    '''
    
    if 'spi' not in extra or extra['spi'] is None: raise ValueError, "Bug in code, spider failed to launch"
    _logger.debug("Processing: %s"%filename)
    id = spider_utility.spider_id(filename, id_len)
    spider_utility.update_spider_files(extra, id, 'output_pow', 'output_roo', 'output_ctf', 'output_mic')  
    _logger.debug("create power spec")
    power_spec, powm = create_powerspectra(filename, **extra)
    _logger.debug("rotational average")
    rotational_average(power_spec, **extra) #ro_arr = 
    _logger.debug("estimate defocus")
    try:
        ang, mag, defocus, overdef, cutoff, unused = extra['spi'].tf_ed(power_spec, outputfile=extra['output_ctf'], **extra)
    except spider.SpiderCrashed:
        _logger.warn("SPIDER crashed - attempting to restart - try increasing the padding and or window size")
        extra['spi'].relaunch()
        ang, mag, defocus, overdef, cutoff, unused = 0, 0, 0, 0, 0, 0
    except:
        _logger.error("Failed to estimate defocus for %s"%filename)
        raise

    return filename, numpy.asarray([id, defocus, ang, mag, cutoff])

def rotational_average(power_spec, spi, output_roo, use_2d=True, **extra):
    '''Compute the rotation average of the power spectra and store as an ndarray
    
    :Parameters:
    
    power_spec : str
                Input filename for power spectra
    spi : spider.Session
          Current SPIDER session
    output_roo : str
                 Output file for rotational average
    use_2d : bool
             Assume 2D power spectra, if true
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    vals : ndarray
           Rotational average of power spectra
    '''
    
    #ma = ndimage_utility.mean_azimuthal(img)[:img.shape[0]/2]
    window_size, = spi.fi_h(power_spec, ('NSAM', ))
    rot_avg = spi.ro(power_spec)
    spi.de(output_roo)
    output_roo = os.path.splitext(output_roo)[0]
    spi.li_d(rot_avg, 'R', 1, outputfile=output_roo+"_old", use_2d=use_2d)
    ro_arr = numpy.asarray(format.read(output_roo+"_old"+ "." + spi.dataext, numeric=True, header="id,amplitude,pixel,a,b"))[:, 1:]
    if ro_arr.shape[1]!=4:
        _logger.error("ROO: %s"%str(ro_arr.shape))
    assert(ro_arr.shape[1]==4)
    ro_arr[:, 2] = ro_arr[:, 0]
    ro_arr[:, 0] = numpy.arange(1, len(ro_arr)+1)
    ro_arr[:, 1] = ro_arr[:, 0] / float(window_size)
    format.write(spi.replace_ext(output_roo), ro_arr[:, :3], default_format=format.spiderdoc, header="index,spatial_freq,amplitude".split(','))
    return ro_arr[:, 1:3]

def create_powerspectra(filename, spi, use_powerspec=False, use_8bit=None, pad=2, du_nstd=[], du_type=3, window_size=256, x_overlap=50, offset=50, output_pow=None, bin_factor=None, invert=None, output_mic=None, bin_micrograph=None, **extra):
    ''' Calculate the power spectra from the given file
    
    :Parameters:
    
    filename : str
               Input filename for the micrograph
    spi : spider.Session
          Current SPIDER session
    use_powerspec : bool
                    Set True if the input file is a power spectra
    pad : int
          Number of times to pad window before Fourier transform; if pad > window-size, then size of padded image (0 means none)
    du_nstd : list
              List of number of standard deviations for dedusting
    du_type : int
              Type of dedusting: (1) BOTTOM, (2) TOP, (3) BOTH SIDES: 3
    window_size : int
                  Size of the window to be cropped from the micrograph for the power spectra (Default: 500)
    x_overlap : int
                Percent overlap in the x-direction (Default: 50)
    offset : int
             Offset from the edge of the micrograph
    output_pow : str, optional
                 Output file for power spectra
    bin_factor : int
                 Decimation factor of the micrograph
    invert : bool
             Invert the contrast of the micrograph
    output_mic : str
                 Output filename for decimated micrograph
    bin_micrograph : float
                     Decimation factor for decimated micrograph
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    power_sec : spider_var
                In-core reference to power spectra image
    '''
    
    if not use_powerspec:       
        try:
            image_count = ndimage_file.count_images(filename)
        except:
            _logger.error("Host: %s"%mpi_utility.hostname())
            raise
        if image_count > 1:
            rwin = ndimage_file.iter_images(filename)
            rwin = itertools.imap(prepare_micrograph, rwin, bin_factor, invert)
        else:
            mic = prepare_micrograph(ndimage_file.read_image(filename), bin_factor, invert)
            if output_mic != "" and bin_micrograph > 0:
                fac = bin_micrograph/bin_factor
                if fac > 1: img = eman2_utility.decimate(mic, fac)
                if use_8bit:
                    img = ndimage_utility.histeq(ndimage_utility.replace_outlier(img, 2.5))
                    img = ndimage_utility.normalize_min_max(img)*255
                    img = img.astype(numpy.uint8)
                    ndimage_file.mrc.write_image(os.path.splitext(output_mic)[0]+".mrc", img)
                else:
                    ndimage_file.spider_writer.write_image(output_mic, img)
            #window_size /= bin_factor
            x_overlap_norm = 100.0 / (100-x_overlap)
            step = max(1, window_size/x_overlap_norm)
            rwin = ndimage_utility.rolling_window(mic[offset:mic.shape[0]-offset, offset:mic.shape[1]-offset], (window_size, window_size), (step, step))
            try:
                rwin = rwin.reshape((rwin.shape[0]*rwin.shape[1], rwin.shape[2], rwin.shape[3]))
            except:
                _logger.error("%d, %d - %s --- %s"%(window_size, step, str(mic.shape), rwin.shape))
                raise
        npowerspec = ndimage_utility.powerspec_avg(rwin, pad)
        mpowerspec = npowerspec.copy()
        
        
        #remove_line(npowerspec)
        if 1 == 0:
            mask = eman2_utility.model_circle(npowerspec.shape[0]*(4.0/extra['pixel_diameter']), npowerspec.shape[0], npowerspec.shape[1])
            sel = mask*-1+1
            npowerspec[mask.astype(numpy.bool)] = numpy.mean(npowerspec[sel.astype(numpy.bool)])
            ndimage_file.write_image(spi.replace_ext(output_pow), npowerspec)
            power_spec = spi.cp(output_pow)
        else:
            _logger.debug("Writing power spectra to %s"%spi.replace_ext(output_pow))
            ndimage_file.write_image(spi.replace_ext(output_pow), npowerspec)
            if not os.path.exists(spi.replace_ext(output_pow)): raise ValueError, "Bug in code: cannot find power spectra at %s"%spi.replace_ext(output_pow)
            power_spec = spi.cp(output_pow)
            spi.du(power_spec, 3, 3)
        
        
        assert(output_pow != "" and output_pow is not None)
        ndimage_file.write_image(spi.replace_ext(output_pow), mpowerspec)
    else:
        power_spec = spi.cp(filename)
        spi.du(power_spec, 3, 3)
        npowerspec = ndimage_file.read_image(spi.replace_ext(filename))
    return power_spec, npowerspec

def prepare_micrograph(mic, bin_factor, invert):
    ''' Preprocess the micrograph
    
    :Parameters:
    
    mic : array
          Micrograph image
    bin_factor : float
                 Number of times to decimate the micrograph
    invert : bool
             True if the micrograph should be inverted
    
    :Returns:
    
    mic : array
          Preprocessed micrograph
    '''
    
    if bin_factor > 1: mic = eman2_utility.decimate(mic, bin_factor)
    if invert: ndimage_utility.invert(mic, mic)
    return mic

def default_path(filename, output):    
    ''' If the filename is not absolute, then append the path of output file
    
    :Parameters:
    
    filename : str
               Filename for secondary output file
    output : str
             Filename primary output file
    
    :Returns:
    
    filename : str
               Filename for correct location
    '''
    
    if filename == "": return filename
    if not os.path.isabs(filename) and os.path.commonprefix( (filename, output ) ) == "":
        path = os.path.dirname(output)
        if path != "": filename = os.path.join(path, filename)
    return filename

def initialize(files, param):
    # Initialize global parameters for the script
    
    if param['output_pow'] == "": param['output_pow']=os.path.join("pow", "pow_00000")
    if param['output_roo'] == "": param['output_roo']=os.path.join("roo", "roo_00000")
    if param['output_ctf'] == "": param['output_ctf']=os.path.join("ctf", "ctf_00000")
    
    if os.path.dirname(param['output_pow']) == 'pow':
        param['output_pow'] = default_path(param['output_pow'], param['output'])
    param['output_roo'] = default_path(param['output_roo'], os.path.dirname(param['output_pow']))
    param['output_ctf'] = default_path(param['output_ctf'],  os.path.dirname(param['output_pow']))
    param['output_mic'] = default_path(param['output_mic'],  os.path.dirname(param['output_pow']))
    if mpi_utility.is_root(**param):
        try: os.makedirs(os.path.dirname(param['output_pow'])) 
        except: pass
        try: os.makedirs(os.path.dirname(param['output_roo'])) 
        except: pass
        try: os.makedirs(os.path.dirname(param['output_ctf'])) 
        except: pass
    mpi_utility.barrier(**param)
    param['spi'] = spider.open_session(files, **param)
    if not os.path.exists(param['spi'].replace_ext(param['param_file'])):
        spider_params.read(param['param_file'], param)
    else:
        spider_params.read(param['spi'].replace_ext(param['param_file']), param)
    if os.path.splitext(param['output'])[1]!='.emx':
        param['output'] = param['spi'].replace_ext(param['output'])
    param['output_pow'] = param['spi'].replace_ext(param['output_pow'])
    param['output_roo'] = param['spi'].replace_ext(param['output_roo'])
    param['output_ctf'] = param['spi'].replace_ext(param['output_ctf'])
    param['output_mic'] = param['spi'].replace_ext(param['output_mic'])
    
    if len(files) > 1 and param['worker_count'] > 1: 
        param['spi'].close()
        param['spi'] = None
        
    if mpi_utility.is_root(**param):
        try:
            defvals = format.read(param['output'], numeric=True)
        except:
            param['output_offset']=0
        else:
            _logger.info("Restarting from defocus file")
            newvals = []
            if param['use_powerspec']:
                newvals = defvals
                defvals = format_utility.map_object_list(defvals)
                oldfiles = list(files)
                files = []
                for f in oldfiles:
                    if spider_utility.spider_id(f, param['id_len']) not in defvals:
                        files.append(f)
                _logger.info("Restarting on %f files of %f total files"%(len(files), len(oldfiles)))
            else:
                ids = set([spider_utility.spider_id(f, param['id_len']) for f in files])
                for d in defvals:
                    if d.id not in ids:
                        newvals.append(d)
            if len(newvals) > 0:
                format.write(param['output'], newvals, default_format=format.spiderdoc)
            param['output_offset']=len(newvals)
        if os.path.splitext(param['output'])[1] == '.emx':
            fout = open(param['output'], 'w')
            fout.write('<EMX version="1.0">\n')
            fout.close()
    
    if mpi_utility.is_root(**param):
        param['defocus_arr'] = numpy.zeros((len(files), 5))
        _logger.info("Estimating defocus over %d files"%(len(files)))
        _logger.info("Writing power spectra to %s"%param['output_pow'])
        if param['output_mic'] != "": 
            _logger.info("Writing decimated micrographs to %s"%param['output_mic'])
        _logger.info("Writing defocus to %s"%param['output'])
        _logger.info("Bin factor: %f"%param['bin_factor'])
        _logger.info("Padding: %d"%param['pad'])
        _logger.info("Pixel size: %f"%(param['apix']))
        _logger.info("Window size: %d"%(param['window_size']))#/param['bin_factor']))
        if param['invert']:
            _logger.info("Inverting Micrograph - common for CCD")
        if param['bin_factor'] != 1.0:
            if not param['disable_bin']:
                _logger.info("Interpolate micrograph with %f"%param['bin_factor'])
            else:
                _logger.info("No micrograph interpolation")
    return sorted(files)

def init_process(input_files, rank=0, **extra):
    # Initialize a child process
    
    rank = mpi_utility.get_offset(**extra)
    param = {}
    _logger.debug("Opening process specific spider: %d"%rank)
    param['spi'] = spider.open_session(input_files, rank=rank, **extra)
    return param

'''

df1 = defocus + astig/2
df2 = defocus - astig/2
if astig < 0
ang = astig + 45


defocus = (DFMID1 + DFMID2)/2;
astig = (DFMID2 - DFMID1);
angle_astig = ANGAST - 45;
if (astig < 0) {
astig = -astig;
angle_astig = angle_astig + 90;
}
'''

def reduce_all(filename, file_completed, file_count, output, defocus_arr, output_offset=0, **extra):
    # Process each input file in the main thread (for multi-threaded code) 38.25 - 3:45
    
    filename, defocus_vals = filename
    
    if os.path.splitext(output)[1] == '.emx':
        fout = open(output, 'a')
        defocusU = defocus_vals[1] + defocus_vals[3]/2
        defocusV = defocus_vals[1] - defocus_vals[3]/2
        defocusUAngle = numpy.mod(defocus_vals[2]-45.0, 180.0)
        fout.write(
        '''
        <micrograph fileName="%s">
        <defocusU unit="nm">%f</defocusU>
        <defocusV unit="nm">%f</defocusV>
        <defocusUAngle unit="deg">%f</defocusUAngle>
        </micrograph>
        '''%(os.path.basename(filename), defocusU/10.0, defocusV/10.0, defocusUAngle))
        fout.close()
    else:
        try:
            defocus_arr[file_completed-1, :]=defocus_vals
        except:
            _logger.error("%d > %d"%(file_completed, len(defocus_arr)))
            raise
        mode = 'a' if (file_completed+output_offset) > 1 else 'w'
        format.write(output, defocus_vals.reshape((1, defocus_vals.shape[0])), format=format.spiderdoc, 
                     header="id,defocus,astig_ang,astig_mag,cutoff_freq".split(','), mode=mode, write_offset=file_completed+output_offset)
    return filename

def finalize(files, defocus_arr, output, output_pow, output_roo, output_ctf, summary=False, **extra):
    # Finalize global parameters for the script
    
    if os.path.splitext(output)[1] == '.emx':
        fout = open(output, 'a')
        fout.write('</EMX>\n')
        fout.close()
    if len(files) > 0:
        plot_histogram(output, defocus_arr[:, 1], 'Defocus')
        plot_histogram(output, defocus_arr[:, 3], 'Astigmatism')
        
        if summary:
            idx = numpy.argsort(defocus_arr[:, 1])
            for i, j in enumerate(idx):
                id =  int(defocus_arr[j, 0])
                roo = numpy.asarray(format.read(output_roo, numeric=True, spiderid=id))
                ctf = numpy.asarray(format.read(output_ctf, numeric=True, spiderid=id))
                pow = ndimage_file.read_image(spider_utility.spider_filename(output_pow, id))
                remove_line(pow)
                pow = label_image(pow, defocus_arr[j], roo[:, 2:], ctf[:, 3], **extra)
                ndimage_file.write_image(format_utility.add_prefix(output, 'stack'), pow, int(i))
    
    _logger.info("Completed")
    
def remove_line(img):
    ''' Remove the line artifact in the center of the power spec
    
    877-242-7372
    1446237624
    '''
    
    m = img.shape[0]/2
    b, e = m-3, m+4
    #idx = range(b-2,b) + range(e,e+2)
    #img[b:e, :] = img[e+5, :] #numpy.mean(img[idx, :], axis=0)
    img[:, b:e] = img[:, e+5][:, numpy.newaxis] #numpy.mean(img[:, idx], axis=1)[:, numpy.newaxis]
    return img

def label_image(img, label, roo, ctf, pixel_diameter, dpi=72, **extra):
    ''' Create a labeled power spectra for display
    '''
    
    pylab = plotting.pylab
    if pylab is None: return img
    fig = pylab.figure(dpi=dpi, facecolor='white')
    ax = pylab.axes(frameon=False)
    newax = ax.twinx()
    
    start = img.shape[0]*(8.0/pixel_diameter)
    start = min_freq(ctf[start:])+start
    mask = eman2_utility.model_circle(start, img.shape[0], img.shape[1])
    sel = mask*-1+1
    img[mask.astype(numpy.bool)] = numpy.mean(img[sel.astype(numpy.bool)])
    ax.imshow(img, cmap=pylab.cm.gray)
    ax.text(3, 15, r'%d - %.2f (%.2f)'%(label[0], label[1], label[3]), color='white')  
    
    #freq = numpy.arange(len(roo))+len(roo)
    #newax.plot(freq[start:], roo[start:, 1], c='w')
    freq = numpy.arange(len(roo))[:len(roo)-start+1]
    newax.plot(freq, ctf[start:][::-1], c='w')  
    val = numpy.max(numpy.abs(ctf[start:]))*4
    newax.set_ylim(-val, val)
    
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    newax.get_yaxis().tick_left()
    newax.axes.get_yaxis().set_visible(False)
    pylab.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    
    return plotting.save_as_image(fig)

def min_freq(roo):
    '''
    '''
    
    cut = numpy.median(numpy.abs(roo)) + analysis.robust_sigma(numpy.abs(roo))*2
    sel = numpy.abs(roo) < cut
    off = numpy.argwhere(sel).squeeze()[0]
    return off

def plot_scatter(output, x, x_label, y, y_label, dpi=72):
    ''' Plot a histogram of the distribution
    '''
    
    pylab=plotting.pylab
    if pylab is None: return
    
    fig = pylab.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    
    '''
    index = select[plotting.nonoverlapping_subset(ax, eigs[select, 0], eigs[select, 1], radius, 100)].ravel()
    iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, label[index].squeeze()))
    plotting.plot_images(fig, iter_single_images, eigs[index, 0], eigs[index, 1], image_size, radius)
    '''
    
    fig.savefig(format_utility.new_filename(output, x_label.lower().replace(' ', '_')+"_"+y_label.lower().replace(' ', '_')+"_", ext="png"), dpi=dpi)
    
def plot_histogram(output, vals, x_label, th=None, dpi=72):
    ''' Plot a histogram of the distribution
    '''
    
    pylab=plotting.pylab
    if pylab is None: return
    
    fig = pylab.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    vals = ax.hist(vals, bins=numpy.sqrt(len(vals)))
    if th is not None:
        h = pylab.gca().get_ylim()[1]
        pylab.plot((th, th), (0, h))
    pylab.xlabel(x_label)
    pylab.ylabel('Number of Micrographs')
    
    '''
    index = select[plotting.nonoverlapping_subset(ax, eigs[select, 0], eigs[select, 1], radius, 100)].ravel()
    iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, label[index].squeeze()))
    plotting.plot_images(fig, iter_single_images, eigs[index, 0], eigs[index, 1], image_size, radius)
    '''
    
    fig.savefig(format_utility.new_filename(output, x_label.lower().replace(' ', '_')+"_", ext="png"), dpi=dpi)

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import setup_options_from_doc, OptionGroup
    
    pgroup.add_option("-i", input_files=[], help="List of input filenames containing micrographs, window stacks or power spectra", required_file=True, gui=dict(filetype="file-list"))
    pgroup.add_option("-o", output="",      help="Output filename for defocus file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
    spider_params.setup_options(parser, pgroup, True)
    
    group = OptionGroup(parser, "Additional", "Options to customize defocus estimation", group_order=0,  id=__name__)
    group.add_option("",   disable_bin=False,                              help="Disable micrograph decimation")
    group.add_option("",   output_pow=os.path.join("pow", "pow_00000"),    help="Filename for output power spectra", gui=dict(filetype="save"))
    group.add_option("",   output_roo=os.path.join("roo", "roo_00000"),    help="Filename for output rotational average", gui=dict(filetype="save"), dependent=False)
    group.add_option("",   output_ctf=os.path.join("ctf", "ctf_00000"),    help="Filename for output CTF curve", gui=dict(filetype="save"), dependent=False)
    group.add_option("",   output_mic="",                                  help="Filename for output for decimated micrograph", gui=dict(filetype="save"), dependent=False)
    group.add_option("",   inner_radius=5,                                 help="Inner mask size for power spectra enhancement")
    group.add_option("",   invert=False,                                   help="Invert the contrast of CCD micrographs")
    group.add_option("",   bin_micrograph=8.0,                             help="Number of times to decimate the micrograph - for micrograph selection (not used for defocus estimation)")
    group.add_option("",   summary=False,                                  help="Write out a summary for the power spectra")
    group.add_option("",  use_8bit=False,                                   help="Write decimate micrograph as 8-bit mrc")
    pgroup.add_option_group(group)
    
    setup_options_from_doc(parser, create_powerspectra, group=pgroup)# classes=spider.Session, for_window_in_micrograph
    if main_option:
        setup_options_from_doc(parser, spider.open_session)
        parser.change_default(thread_count=4, log_level=3, bin_factor=2)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if options.window_size == 0: raise OptionValueError, "Window size must be greater than zero"
    spider_params.check_options(options)
    if main_option:
        if not spider_utility.test_valid_spider_input(options.input_files):
            raise OptionValueError, "Multiple input files must have numeric suffix, e.g. vol0001.spi"
        for f in options.input_files:
            if not ndimage_file.is_readable(f): 
                raise OptionValueError, "Unrecognized image format for input-file: %s \n Check if you have permission to access this file and this file is in an acceptable format"%f
        if ndimage_file.is_spider_format(options.input_files[0]) and options.data_ext == "":
            raise OptionValueError, "You must set --data-ext when the input file is not in SPIDER format"

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Estimate the defocus of a set of micrographs or particle stacks
                        
                        http://guam/vispider/vispider/manual.html#module-vispider.batch.defocus
                        
                        Note: If this script crashes or hangs, try increasing the padding and or window size.
                        
                        $ spi-defocus mic_*.ter -p params.ter -o defocus.ter
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=True,
        use_version = True,
        max_filename_len = 78,
    )
if __name__ == "__main__": main()


