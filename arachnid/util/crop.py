'''Extracting windows from a micrograph (Cropping)

This script (`ara-crop`) extracts particles from a micrograph based on coordinates (which may
have come from AutoPicker).

It performs the following preprocessing on the micrograph:

    - High pass filter with cutoff = sigma / window_size (Default sigma = 1, if sigma = 0, then high pass filtering is disabled)
    - Decimation (Default bin-factor = 0, no decimation)
    - Contrast inversion (Default invert = False)

In addition, it performs the following preprocessing on the windows:
    
    - Ramp filtering removing changes in illumination
    - Removing bad pixels from dust or errant electrons (Default clamp-window 5, if clamp-window = 0 then disable clamping)
    - Histogram matching to a noise window
    - Normalization to mean=0, variance=1 outside the particle radius (Default on, disable-normalize will disable this step)

Tips
====

 #. Input filenames: To specifiy multiple files, either use a selection file `--selection-doc sel01.spi` with a single input file mic_00001.spi or use a single star mic_*.spi to
    replace the number.
 
 #. All other filenames: Must follow the SPIDER format with a number before the extension, e.g. mic_00001.spi. Output files just require the number of digits: `--output sndc_0000.spi`

 #. CCD micrographs - Use the `--invert` parameter to invert CCD micrographs. In this way, you can work directly with the original TIFF
 
 #. Decimation - Use the `--bin-factor` parameter to reduce the size of the micrograph for more efficient processing. If the micrographs are already decimated but not the coordinates,
    then use both `--bin-factor` and `--disable-bin` to change the coordinates but not the micrographs.
 

Running Cropping
================

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Create a configuration file
    
    $ ara-crop > crop.cfg
    
    # Alternative: Create configuration file
    
    $ ara-crop -i Micrographs/mic_00001.spi -r 55 -s coords/sndc_0000.spi -o "" > crop.cfg
    
    # Edit configuration file
    
    $ vi crop.cfg      # -- or
    $ kwrite crop.cfg  # -- or
    $ pico crop.cfg
    
    # Run Program
    
    $ ara-crop -c crop.cfg
    
    # Alternative: Run in background (Recommended when running on a remote machine)
    
    $ nohup ara-crop -c crop.cfg > crop.log &


Critical Options
================

.. program:: ara-crop

.. option:: -i <filename1,filename2>, --input-files <filename1,filename2>, filename1 filename
    
    List of input filenames containing micrographs
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`

.. option:: -o <str>, --output <str>
    
    Output filename for window stack with correct number of digits (e.g. sndc_0000.spi)

.. option:: -s <str>, --coordinate-file <str>
    
    Filename to a set of particle coordinates
    
.. option:: -r <int>, --pixel-radius <int>
    
    Size of your particle in pixels. If you decimate with `--bin-factor` give the undecimated value
    
.. option:: --window <float>
    
    Size of your window in pixels or multiplier (if less than the diameter). If you decimate with `--bin-factor` give the undecimated value


Useful Options
==============

.. program:: ara-crop

.. option:: -w <int>, --worker-count <int>
    
    Set the number of micrographs to process in parallel (keep in mind memory and processor restrictions)
    
.. option:: --invert
    
    Invert the contrast of CCD micrographs
    
.. option:: --bin-factor <int>
    
    Decimate the micrograph

.. option:: --disable-enhance
    
    Disable window post-processing: ramp, contrast enhancement on the windows

.. option:: --disable-bin
    
    Disable micrograph decimation, but still decimate coordinates, pixel-radius and window-size

.. option:: --disable-normalize
    
    Disable XMIPP normalization

.. option:: --clamp-window <float>
    
    Number of standard deviations to replace extreme values using a Gaussian distribution (0 to disable)

.. option:: --sigma <float>
    
    Highpass factor: 1 or 2 where 1/window size or 2/window size (0 to disable)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by MPI-enabled scripts... <mpi-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`

.. todo:: replace image_reader with ndimage_format

.. Created on Nov 27, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
ndimage_file=None
from ..core.image import eman2_utility, ndimage_utility, ndimage_file # - replace image_reader and writer
from ..core.image import reader as image_reader, writer as image_writer
from ..core.metadata import spider_utility, format_utility, format, spider_params
from ..core.image.formats import mrc as mrc_file
from ..core.parallel import mpi_utility
import numpy, os, logging, psutil

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, id_len=0, single=False, **extra):
    '''Crop a set of particles from a micrograph file with the specified
    coordinate file and write particles to an image stack.
    
    :Parameters:
    
    filename : str
               Input filename
    id_len : int
             Maximum length of ID
    extra : dict
            Unused keyword arguments
                
    :Returns:
        
    val : string
          Current filename
    '''
    
    if isinstance(filename, tuple):
        tot = len(filename[1])
        fid = filename[0]
    else:
        try:
            tot = mrc_file.count_images(filename)
        except: tot = ndimage_file.count_images(filename)
        fid = spider_utility.spider_id(filename, id_len)
    spider_utility.update_spider_files(extra, fid, 'selection_doc', 'output', 'coordinate_file', 'frame_align')
    try: coords = read_coordinates(**extra)
    except: 
        _logger.exception("Ignore this")
        return filename, 0, os.getpid()
    
    noise=extra['noise']
    radius, offset, bin_factor, tmp = init_param(**extra)
    norm_mask=tmp*-1+1
    emdata = eman2_utility.utilities.model_blank(offset*2, offset*2)
    npdata = eman2_utility.em2numpy(emdata)
    
    #if extra['experimental']: return filename, len(coords)
    
    output=extra['output']
    if extra['frame_align'] != "":
        if not os.path.exists(extra['frame_align']):
            _logger.warn("No translation file, skipping %s"%str(filename))
            return filename, 0, os.getpid()
        align = format.read(extra['frame_align'], numeric=True)
    else: align=None
    coords_orig=None
    for i in xrange(tot):
        frame = spider_utility.spider_id(filename[1][i]) if isinstance(filename, tuple) else i+1
        _logger.info("Cropping from movie %d frame %d - %d of %d"%(fid, frame, i, tot))
        mic = read_micrograph(filename, i, **extra)    
        # translate
        npmic = eman2_utility.em2numpy(mic) if eman2_utility.is_em(mic) else mic
        bin_factor = extra['bin_factor']
        if tot > 1: 
            output = format_utility.add_prefix(extra['output'], 'frame_%d_'%(frame))
            if extra['experimental']:
                if coords_orig is None: 
                    coords_orig=numpy.asarray(coords).copy()
                coords = coords_orig[:, 1:3] - (align[i].dx/bin_factor, align[i].dy/bin_factor)
            else:
                npmic[:] = eman2_utility.fshift(npmic, align[i].dx/bin_factor, align[i].dy/bin_factor)
        if i == 0:
            test_coordinates(npmic, coords, bin_factor)
            
        _logger.info("Extract %d windows from movie %d frame %d - %d of %d"%(len(coords), fid, frame, i, tot))
        for index, win in enumerate(ndimage_utility.for_each_window(npmic, coords, offset*2, bin_factor)):
            npdata[:, :] = win
            
            #m = numpy.mean(npdata*npmask)
            #s = numpy.std(npdata*npmask)
            #_logger.error("0. index=%d, mean=%f, std=%f"%(index, m, s))
            win = enhance_window(emdata, noise, norm_mask, **extra)
            win.update()
            if win.get_attr("minimum") == win.get_attr("maximum"):
                coord = coords[index]
                x, y = (coord.x, coord.y) if hasattr(coord, 'x') else (coord[1], coord[2])
                _logger.warn("Window %d at coordinates %d,%d has an issue - clamp_window may need to be increased"%(index+1, x, y))
            if ndimage_file is not None:
                ndimage_file.write_image(output, emdata, index)
            else:
                image_writer.write_image(output, emdata, index)
            if single: break
            _logger.info("Extract %d windows from movie %d frame %d - %d of %d - finished"%(len(coords), fid, frame, i, tot))
    return filename, len(coords), os.getpid()

def test_coordinates(npmic, coords, bin_factor):
    ''' Test if the coordinates cover the micrograph properly
    
    :Parameters:
    
    npmic : array
            Array 2-D for micrograph
    coords : list
             List of namedtuple coordinates
    bin_factor : float
                 Decimation factor for the coordinates
    '''
    
    coords, header = format_utility.tuple2numpy(coords)
    try: x = header.index('x')
    except: raise ValueError, "Coordinate file does not have a label for the x-column"
    try: y = header.index('y')
    except: raise ValueError, "Coordinate file does not have a label for the y-column"
    if bin_factor > 0.0: 
        coords[:, x] /= bin_factor
        coords[:, y] /= bin_factor
    if numpy.min(coords[:, x]) < 0: raise ValueError, "Invalid x-coordate: %d"%numpy.min(coords[:, x])
    if numpy.min(coords[:, y]) < 0: raise ValueError, "Invalid y-coordate: %d"%numpy.min(coords[:, y])
    
    if numpy.max(coords[:, x]) > npmic.shape[1]: 
        raise ValueError, "The x-coordate has left the micrograph - consider changing --bin-factor: %d"%numpy.max(coords[:, x])
    if numpy.max(coords[:, y]) > npmic.shape[0]: 
        raise ValueError, "The y-coordate has left the micrograph - consider changing --bin-factor: %d"%numpy.max(coords[:, y])
    
    if numpy.max(coords[:, x])*2 < npmic.shape[1]: 
        _logger.warn("The maximum x-coordate is less than twice the size of the micrograph width - consider changing --bin-factor: %d"%numpy.max(coords[:, x]))
    if numpy.max(coords[:, y])*2 < npmic.shape[0]: 
        _logger.warn("The maximum y-coordate is less than twice the size of the micrograph height - consider changing --bin-factor: "%numpy.max(coords[:, y]))

def read_micrograph(filename, index=0, emdata=None, bin_factor=1.0, sigma=1.0, flip=False, disable_bin=False, invert=False, experimental=False, **extra):
    ''' Read a micrograph from a file and perform preprocessing
    
    :Parameters:
        
    filename : str
               Filename for micrograph
    emdata : EMData
             Reuse allocated memory
    bin_factor : float
                Downsampling factor
    sigma : float
            Gaussian highpass filtering factor (sigma/window)
    disable_bin : bool    
                  If True, do not downsample the micrograph
    invert : bool
             If True, invert the contrast of the micrograph (CCD Data)
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
        
    mic : EMData
          Micrograph image
    '''
    
    if isinstance(filename, tuple):
        filename = filename[1][index]
        index=None
    
    offset = init_param(bin_factor=bin_factor, **extra)[1]
    if ndimage_file is not None:
        mic = ndimage_file.read_image(filename, index, cache=emdata)
    else:
        mic = image_reader.read_image(filename, index, emdata=emdata)
    if flip:
        emmic=eman2_utility.numpy2em(mic)
        emmic.process_inplace("xform.flip",{"axis":"y"})
        mic[:]=eman2_utility.em2numpy(emmic)
    if bin_factor > 1.0 and not disable_bin: 
        mic = eman2_utility.decimate(mic, bin_factor)
    if invert:
        _logger.debug("Invert micrograph")
        if eman2_utility.is_em(mic):
            npmic = eman2_utility.em2numpy(mic)
            ndimage_utility.invert(npmic, npmic)
        else: ndimage_utility.invert(mic, mic)
    if sigma > 0.0:
        mic = eman2_utility.gaussian_high_pass(mic, sigma/(2.0*offset), True)
    if not eman2_utility.is_em(mic) and not experimental:
        mic = eman2_utility.numpy2em(mic)
    if experimental:
        assert(not eman2_utility.is_em(mic))
    return mic
            
def generate_noise(filename, noise="", output="", noise_stack=True, experimental=False, **extra):
    ''' Automatically generate a stack of noise windows and by default choose the first
    
    :Parameters:
        
    filename : str
               Input micrograph filename
    noise : str
            Input filename for existing noise window
    output : str
             Output filename for window stacks
    noise_stack : bool
                  If True, write out full noise stack rather than single window
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
        
    noise_win : EMData
                Noise window
    '''
    
    if noise != "": 
        if ndimage_file is not None:
            return eman2_utility.numpy2em(ndimage_file.read_image(noise))
        return image_reader.read_image(noise)
    noise_file = format_utility.add_prefix(output, "noise_")
    if os.path.exists(noise_file):
        _logger.warn("Found cached noise file: %s - delete if you want to regenerate"%noise_file)
        if ndimage_file is not None:
            img = ndimage_file.read_image(noise_file)
            return eman2_utility.numpy2em(img)
        return image_reader.read_image(noise_file)
    
    try:
        tot = mrc_file.count_images(filename)
    except: tot = ndimage_file.count_images(filename)
    if tot > 1:
        return None
    mic = read_micrograph(filename, **extra)
    rad, offset = init_param(**extra)[:2]
    width = offset*2
    #template = eman2_utility.utilities.model_blank(width, width)
    
    template = eman2_utility.utilities.model_circle(rad, width, width)
    template.process_inplace("normalize.mask", {"mask": template, "no_sigma": True})

    # Define noise distribution
    if not eman2_utility.is_em(mic) and not experimental:
        mic = eman2_utility.numpy2em(mic)
    
    if experimental:
        etemplate = template
        template = eman2_utility.em2numpy(etemplate)
        mic2 = eman2_utility.em2numpy(mic) if eman2_utility.is_em(mic) else mic
        cc_map = ndimage_utility.cross_correlate(mic2, template)
        numpy.fabs(cc_map, cc_map)
        cc_map -= float(numpy.max(cc_map))
        cc_map *= -1
        peak1 = ndimage_utility.find_peaks_fast(cc_map, offset)
        peak1 = numpy.asarray(peak1).squeeze()
    else:
        cc_map = mic.calc_ccf(template)
        cc_map.process_inplace("xform.phaseorigin.tocenter")
        np = eman2_utility.em2numpy(cc_map)
        numpy.fabs(np, np)
        cc_map.update()
        cc_map -= float(numpy.max(np))
        cc_map *= -1
        peak1 = cc_map.peak_ccf(offset)
        peak1 = numpy.asarray(peak1).reshape((len(peak1)/3, 3))
    index = numpy.argsort(peak1[:,0])[::-1]
    peak1 = peak1[index].copy().squeeze()
    
    best = (1e20, None)
    emdata = eman2_utility.utilities.model_blank(offset*2, offset*2)
    npdata = eman2_utility.em2numpy(emdata)
    npmic = eman2_utility.em2numpy(mic)
    for i, win in enumerate(ndimage_utility.for_each_window(npmic, peak1, offset*2, 1.0)):
        npdata[:, :] = win
        win = enhance_window(emdata, None, None, **extra)
        std = numpy.std(eman2_utility.em2numpy(win))
        if std < best[0]: best = (std, win)
        if noise_stack and i < 11: 
            if ndimage_file is not None:
                ndimage_file.write_image(noise_file, win, i)
            else:
                image_writer.write_image(noise_file, win, i)
    noise_win = best[1]
    if not noise_stack:
        if ndimage_file is not None:
            ndimage_file.write_image(noise_file, noise_win)
        else:
            image_writer.write_image(noise_file, noise_win)
    return noise_win

def init_param(pixel_radius, pixel_diameter=0.0, window=1.0, bin_factor=1.0, **extra):
    ''' Ensure all parameters have the proper scale and create a mask
    
    :Parameters:
        
    pixel_radius : float
                  Radius of the particle
    pixel_diameter : int
                     Diameter of the particle
    window : float
             Size of the window (if less than particle diameter, assumed to be multipler)
    bin_factor : float
                 Decimation factor
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    rad : float
          Radius of particle scaled by bin_factor
    offset : int
             Half-width of the window
    bin_factor : float
                 Decimation factor
    mask : EMData
           Disk mask with `radius` that keeps data in the disk
    '''
    
    if pixel_diameter > 0:
        rad = int( float(pixel_diameter) / 2 )
        offset = int( window / 2.0 )
    else:
        rad = int( pixel_radius / float(bin_factor) )
        offset = int( window / (float(bin_factor)*2.0) )
    if window == 1.0: window = 1.4
    if window < (2*rad): offset = int(window*rad)
    width = offset*2
    mask = eman2_utility.utilities.model_circle(rad, width, width)
    _logger.debug("Radius: %d | Window: %d"%(rad, offset*2))
    return rad, offset, bin_factor, mask

def enhance_window(win, noise_win=None, norm_mask=None, mask=None, clamp_window=0.0, disable_enhance=False, disable_normalize=False, use_vst=False, experimental=False, **extra):
    '''Enhance the window with a set of filtering and normalization routines
    
    :Parameters:
        
    win : EMData
          Input raw window
    noise_win : EMData
                Noise window
    norm_mask : EMData
                Disk mask with `radius` that keeps data outside the disk
    mask : EMData
           Disk mask with `radius` that keeps data inside the disk
    clamp_window : float
                   Number of standard deviations to replace extreme values using a Gaussian distribution
    disable_enhance : bool
                      Disable all enhancement
    disable_normalize : bool
                       Disable normalization
    use_vst : bool
              Apply the VST transform (if set True)
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
        
    win : EMData
          Enhanced window
    '''
    
    if not disable_enhance:
        np = eman2_utility.em2numpy(win)
        #assert(noise_win.get_attr("minimum")<noise_win.get_attr("maximum"))
        _logger.debug("Removing ramp from window")
        
        eman2_utility.ramp(win)
        if use_vst: ndimage_utility.vst(np, np)
        if clamp_window > 0: 
            _logger.debug("Removing outlier pixels")
            ndimage_utility.replace_outlier(np, clamp_window, out=np)
        if win.get_attr("minimum") == win.get_attr("maximum"): return win
    
        if noise_win is not None: 
            _logger.debug("Improving contrast with histogram fitting: (%f,%f,%f,%f) (%f,%f,%f,%f)"%(win.get_attr("mean"), win.get_attr("sigma"), win.get_attr("minimum"), win.get_attr("maximum"), noise_win.get_attr("mean"), noise_win.get_attr("sigma"), noise_win.get_attr("minimum"), noise_win.get_attr("maximum")))
            if not experimental:
                _logger.debug("using old")
                win = eman2_utility.histfit(win, mask, noise_win)
            else:
                nref = eman2_utility.em2numpy(noise_win) if eman2_utility.is_em(noise_win) else noise_win
                nmask = eman2_utility.em2numpy(mask) if eman2_utility.is_em(mask) else mask
                ndimage_utility.histogram_match(np, nmask, nref, out=np)
        if not disable_normalize and norm_mask is not None:
            #if mask is None:
            #    mask = eman2_utility.model_circle(radius, img.shape[0], img.shape[1])*-1+1
            #img = ndimage_utility.normalize_standard(img, mask)
            #win.process_inplace("normalize.mask", {"mask": norm_mask, "no_sigma": 0})
            ndimage_utility.normalize_standard(np, eman2_utility.em2numpy(norm_mask), out=np)
        _logger.debug("Finished Enhancment")
    return win

def read_coordinates(coordinate_file, selection_doc="", **extra):
    ''' Read a coordinate file (use `selection_doc` to select a subset if specified)
    
    :Parameters:
        
    coordinate_file : str
                      Filename for input coordinate file
    selection_doc : str
                    Filename for input selection file
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
        
    coords : list
             List of coordinates (namedtuple)
    '''
    
    try:
        select = format.read(selection_doc, numeric=True) if selection_doc != "" and spider_utility.is_spider_filename(selection_doc) else None
    except:
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.exception("Skipping: %s"%selection_doc)
        else:
            _logger.warn("Cannot find selection file: %s"%selection_doc)
        raise
    try:
        blobs = format.read(coordinate_file, numeric=True)
    except:
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.exception("Skipping: %s"%coordinate_file)
        else:
            _logger.warn("Cannot find coordinate file: %s"%coordinate_file)
        raise
    if select is not None and 'id' in select[0]._fields:
        tmp = blobs
        blobs = []
        if 'select' in select[0]._fields:
            for s in select:
                if s.select > 0:
                    try:
                        blobs.append(tmp[s.id-1])
                    except:
                        if _logger.isEnabledFor(logging.DEBUG):
                            _logger.exception("%d > %d"%(s.id-1, len(tmp)))
        else:
            for s in select:
                try:
                    blobs.append(tmp[s.id-1])
                except:
                    if _logger.isEnabledFor(logging.DEBUG):
                        _logger.exception("%d > %d"%(s.id-1, len(tmp)))
    return blobs


def init_root(files, param):
    # Initialize global parameters for the script
 
    if mpi_utility.is_root(**param):
        try: tot = mrc_file.count_images(files[0])
        except: tot = ndimage_file.count_images(files[0])
        if param['frame_align'] != "" and tot == 1:
            files = spider_utility.single_images(files)
            if param['reverse']:
                for f in files:
                    f[1].reverse()
            if param['reverse']:
                _logger.info("Processing example movie in reverse: %s"%str(files[0]))
            else:
                _logger.info("Processing example movie: %s"%str(files[0]))
            
        _logger.info("Processing %d micrographs"%len(files))
    
    files = mpi_utility.broadcast(files, **param)
    return files

def initialize(files, param):
    # Initialize global parameters for the script
    
    radius, offset, bin_factor, param['mask'] = init_param(**param)
    if mpi_utility.is_root(**param):
        _logger.info("Processing %d files"%len(files))
        _logger.info("Pixel radius: %d"%radius)
        _logger.info("Window size: %d"%(offset*2))
        if param['sigma'] > 0: _logger.info("High pass filter: %f"%(param['sigma'] / (offset*2)))
        if param['bin_factor'] > 1 and not param['disable_bin']: _logger.info("Decimate micrograph by %d"%param['bin_factor'])
        if param['invert']: _logger.info("Inverting contrast of the micrograph")
        if not param['disable_enhance']:
            _logger.info("Ramp filter")
            _logger.info("Dedust: %f"%param['clamp_window'])
            _logger.info("Histogram matching")
            if not param['disable_normalize']: _logger.info("Normalize (xmipp style)")
        param['count'] = numpy.zeros(2, dtype=numpy.int)
        if len(files) > 0 and not isinstance(files[0], tuple):
            _logger.info("Extracting windows from movies frames from micrograph stacks")
    
    if mpi_utility.is_root(**param):
        selection_doc = format_utility.parse_header(param['selection_doc'])[0]
        if not spider_utility.is_spider_filename(selection_doc) and os.path.exists(selection_doc):
            select = format.read(param['selection_doc'], numeric=True)
            file_count = len(files)
            if len(select) > 0:
                select = set([s.id for s in select])
                old_files = files
                files = []
                for filename in old_files:
                    id = filename[0] if isinstance(filename, tuple) else spider_utility.spider_id(filename, param['id_len'])
                    if id in select: files.append(filename)
                finished = param['finished']
                param['finished'] = []
                for filename in finished:
                    id = filename[0] if isinstance(filename, tuple) else spider_utility.spider_id(filename, param['id_len'])
                    if id in select:
                        param['finished'].append(filename)
            _logger.info("Assuming %s is a micrograph selection file - found %d micrographs of %d"%(selection_doc, len(files), file_count))
        if isinstance(files[0], tuple):
            tot = len(files[0][1])
        else:
            try:tot = mrc_file.count_images(files[0])
            except: tot = ndimage_file.count_images(files[0])
        for filename in param['finished']:
            if tot > 1:
                id = filename[0] if isinstance(filename, tuple) else filename
                ncoord = len(format.read(param['coordinate_file'], numeric=True, spiderid=id, id_len=param['id_len']))
                # Todo only add frames that require processing
                for i in xrange(tot):
                    frame = spider_utility.spider_id(filename[1][i]) if isinstance(filename, tuple) else i+1
                    output = format_utility.add_prefix(param['output'], 'frame_%d_'%(frame))
                    nimage = ndimage_file.count_images(spider_utility.spider_filename(output, id, param['id_len']))
                    if nimage != ncoord:
                        _logger.info("Found partial stack: %d != %d"%(ncoord, nimage))
                        files.append(filename)
                        break
            else:
                id = filename[0] if isinstance(filename, tuple) else filename
                ncoord = len(format.read(param['coordinate_file'], numeric=True, spiderid=id, id_len=param['id_len']))
                nimage = ndimage_file.count_images(spider_utility.spider_filename(param['output'], id, param['id_len']))
                if nimage != ncoord:
                    _logger.info("Found partial stack: %d != %d"%(ncoord, nimage))
                    files.append(filename)
               
    files = mpi_utility.broadcast(files, **param)
    
    
    if len(files) == 0:
        param['noise']=None
    elif param['noise'] == "":
        if not isinstance(files[0], tuple):
            
            if mpi_utility.is_root(**param):
                param['noise'] = generate_noise(files[0], **param)
            mpi_utility.barrier(**param)
            if mpi_utility.is_client_strict(**param):
                param['noise'] = generate_noise(files[0], **param)
        else: param['noise']=None
        
    else: 
        if ndimage_file is not None:
            param['noise'] =  eman2_utility.numpy2em(ndimage_file.read_image(param['noise']))
        else:
            param['noise'] = image_reader.read_image(param['noise'])
    param['emdata'] = eman2_utility.EMAN2.EMData()
    return files

def reduce_all(val, count, id_len, **extra):
    # Process each input file in the main thread (for multi-threaded code)

    filename, total, pid = val
    count[0]+= total
    count[1]+= 1
    if isinstance(filename, tuple): filename=filename[0]
    return str(filename)+" - %d windows - %d windows in total in %d files - %1f gigs"%(total, count[0], count[1], psutil.Process(pid).get_memory_info().rss/131072.0)

def finalize(files, count, **extra):
    # Finalize global parameters for the script
    _logger.info("Extracted %d windows"%count[0])
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    
    group = OptionGroup(parser, "Cropping", "Options to crop particles from micrographs", id=__name__) #, gui=dict(root=True, stacked="prepComboBox"))
    group.add_option("", disable_enhance=False,    help="Disable window post-processing: ramp, contrast enhancement")
    group.add_option("", disable_bin=False,        help="Disable micrograph decimation")
    group.add_option("", disable_normalize=False,  help="Disable XMIPP normalization")
    group.add_option("", clamp_window=2.5,         help="Number of standard deviations to replace extreme values using a Gaussian distribution (0 to disable)")
    group.add_option("", sigma=1.0,                help="Highpass factor: 1 or 2 where 1/window size or 2/window size (0 to disable)")
    group.add_option("", use_vst=False,            help="Use variance stablizing transform")
    group.add_option("", invert=False,             help="Invert the contrast on the micrograph (important for raw CCD micrographs)")
    group.add_option("", noise="",                 help="Use specified noise file")
    group.add_option("-r", pixel_radius=0,         help="Radius of the expected particle (if default value 0, then overridden by SPIDER params file, `param-file`)")
    group.add_option("",   window=1.0,             help="Size of the output window or multiplicative factor if less than particle diameter (overridden by SPIDER params file, `param-file`)")
    group.add_option("", experimental=False,       help="Use new experimental code for memory management")
    group.add_option("", frame_align="",           help="Translational alignment parameters for individual frames")
    group.add_option("", single=False,             help="Single window (first)")
    group.add_option("", flip=False,                help="Flip micrograph")
    group.add_option("", reverse=False,             help="Reverse for reversied alignment")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[],         help="List of input filenames containing micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",              help="Output filename for window stack with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-s", coordinate_file="",     help="File containing coordinates of objects", gui=dict(filetype="open"), required_file=True)
        pgroup.add_option("-d", selection_doc="",       help="Selection file for a subset of good windows", gui=dict(filetype="open"), required_file=False)
        parser.change_default(log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    if options.bin_factor == 0: raise OptionValueError, "Bin factor cannot be zero"

def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Crop a set of particles from a micrograph
        
                        http://
                         
                        Example: Run from the command line on a single node:
                        
                        $ %prog micrograph.spi -o particle-stack.spi -r 110 -s particle-coords.spi -w 312
                      ''',
        supports_OMP=True,
    )

def dependents(): return [spider_params]

if __name__ == "__main__": main()


