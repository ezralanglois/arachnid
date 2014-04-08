'''Crops windowed particles from a micrograph

This script (`ara-crop`) crops windows containing particles from a micrograph based on coordinates from a particle picking
algorithm such as AutoPicker (`ara-autopick`).

It performs the following preprocessing on the micrograph:

    - High pass filter with cutoff = sigma / window_size (Default sigma = 1, if sigma = 0, then high pass filtering is disabled)
    - Decimation (Default bin-factor = 1, no decimation)
    - Contrast inversion (Default invert = False)

In addition, it performs the following preprocessing on the windows:
    
    - Ramp filtering removing changes in illumination
    - Removing bad pixels from dust or errant electrons (Default clamp-window 5, if clamp-window = 0 then disable clamping)
    - Histogram matching to a noise window
    - Normalization to mean=0, variance=1 outside the particle radius (Default on, disable-normalize will disable this step)

Unless specified, this script automatically finds a noise window (or set of noise windows) from the first micrograph.

Notes
=====

 #. Input filenames: To specifiy multiple files, either use a selection file `--selection-doc sel01.spi` with a single input file mic_00001.spi or use a single star mic_*.spi to
    replace the number.
 
 #. All other filenames: Must follow the SPIDER format with a number before the extension, e.g. mic_00001.spi. Output files just require the number of digits: `--output sndc_0000.spi`

 #. CCD micrographs - Use the `--invert` parameter to invert CCD micrographs. In this way, you can work directly with the original TIFF
 
 #. Decimation - Use the `--bin-factor` parameter to reduce the size of the micrograph for more efficient processing. If the micrographs are already decimated but not the coordinates,
    then use both `--bin-factor` and `--disable-bin` to change the coordinates but not the micrographs.
 

Running the Script
==================

.. sourcecode :: sh
    
    # Create a configuration file
    
    $ ara-crop --create-cfg crop.cfg
    
    # - or -
    
    $ ara-crop -i Micrographs/mic_00001.spi -p params.spi -s coords/sndc_0000.spi -o win/win_0000.spi --create-cfg crop.cfg
    
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

.. option:: --micrograph-files <filename1,filename2>, -i <filename1,filename2>, --input-files <filename1,filename2>, filename1 filename
    
    List of input micrograph filenames
    If you use the parameters `-i` or `--inputfiles` the filenames may be comma or 
    space separated on the command line; they must be comma seperated in a configuration 
    file. Note, these flags are optional for input files; the filenames must be separated 
    by spaces. For a very large number of files (>5000) use `-i "filename*"`

.. option:: --particle-stack <str>, -o <str>, --output <str>
    
    Output filename template for window stack with correct number of digits (e.g. sndc_0000.spi)

.. option:: -s <str>, --coordinate-file <str>
    
    Input filename template containing particle coordinates with correct number of digits (e.g. sndc_0000.spi)

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Filename for SPIDER parameter file describing a Cryo-EM experiment

Enhancement Options
===================

.. program:: ara-crop

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

Movie-mode Options
==================
    
.. option:: --frame-beg <int>
    
    Index of first frame
    
.. option:: --frame-end <int>
    
    Index of last frame
    
.. option:: --frame-align <str>
    
    Translational alignment parameters for individual frames
    
More Options
============
    
.. option:: --selection-file <str>
    
    Selection file for a subset of micrographs or selection file template for subset of good particles
    
.. option:: --bin-factor <FLOAT>
    
    Decimatation factor for the script: changes size of images, coordinates, parameters such as pixel_size or window unless otherwise specified
    
.. option:: --invert
    
    Invert the contrast on the micrograph (usually for raw CCD micrographs)
    
.. option:: --noise <str>
    
    Use specified noise file rather then automatically generate one

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by MPI-enabled scripts... <mpi-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`

.. Created on Nov 27, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.image import ndimage_utility
from ..core.image import ndimage_file
from ..core.image import ndimage_filter
from ..core.image import ndimage_interpolate
from ..core.metadata import spider_utility
from ..core.metadata import format_utility
from ..core.metadata import format
from ..core.metadata import spider_params
from ..core.metadata import selection_utility
from ..core.parallel import mpi_utility
import numpy
import os
import logging
import psutil

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, id_len=0, frame_beg=0, frame_end=0, single_stack=False, **extra):
    '''Crop a set of particles from a micrograph file with the specified
    coordinate file and write particles to an image stack.
    
    :Parameters:
    
        filename : str
                   Input filename
        id_len : int
                 Maximum length of ID
        frame_beg : int
                    First frame to crop
        frame_end : int
                    List frame to crop
        single_stack : bool
                       Write windows to a single stack
        extra : dict
                Unused keyword arguments
                
    :Returns:
            
        val : string
              Current filename
    '''
    
    numpy.seterr('raise')
    if isinstance(filename, tuple):
        tot = len(filename[1])
        fid = filename[0]
    else:
        tot = None
        fid = spider_utility.spider_id(filename, id_len)
    spider_utility.update_spider_files(extra, fid, 'good_file', 'output', 'coordinate_file', 'frame_align')
    try: coords = read_coordinates(**extra)
    except: 
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.exception("Failed to read coordinates file")
        return filename, 0, os.getpid()
    global_selection=extra['global_selection']

    if tot is None:
        try:
            tot = ndimage_file.count_images(filename)
        except:  
            _logger.warn("Skipping: %s - no header"%filename)
            return filename, 0, os.getpid()
        
    
    noise=extra['noise']
    window = extra['window']
    bin_factor = extra['bin_factor']
    
    
    if frame_beg > 0: frame_beg -= 1
    if frame_end < 0: frame_end = tot
    output=extra['output']
    if extra['frame_align'] != "":
        if not os.path.exists(extra['frame_align']):
            _logger.warn("No translation file, skipping %s"%str(filename))
            return filename, 0, os.getpid()
        align = format.read(extra['frame_align'], numeric=True)
        align = format_utility.map_object_list(align)
        if len(align) < (frame_end-frame_beg): 
            _logger.warn("Skipping number of translations is less than frames %d, %d"%(frame_beg, frame_end))
            return filename, 0, os.getpid()
    else: align=None
    
    if tot > 1: _logger.info("Cropping windows from %d frame to %d frame"%(frame_beg,frame_end))
    
    extra['output']=strip_frame_tag(extra['output'])
    
    #for i in xrange(frame_beg,frame_end):
    indexes = range(frame_beg,frame_end) if align is None else [align[i].id for i in xrange(frame_beg,frame_end) ]
    try:
        for j, mic in enumerate(iter_micrographs(filename, indexes, **extra)):
            i = indexes[j]
            frame = spider_utility.spider_id(filename[1][i]) if isinstance(filename, tuple) else i+1
            '''
            if align is not None:
                id=align[i].id
            else:
            '''
            #id=i
            #_logger.info("Cropping from movie %d frame %d - %d of %d"%(fid, frame, id, frame_end))
            #mic = read_micrograph(filename, id, **extra)
            if tot > 1:
                output = format_utility.add_prefix(extra['output'], 'frame_%d_'%(frame))
                if align is not None:
                    mic[:] = ndimage_utility.fourier_shift(mic, -align[i].dx/bin_factor, -align[i].dy/bin_factor)
                #scp /catalina.F30/frames/13nov23c/rawdata/13*en.frames.mrc.bz2
            _logger.info("Extract %d windows from movie %d frame %d - %d of %d"%(len(coords), fid, frame, i, frame_end))
            for index, win in enumerate(ndimage_utility.for_each_window(mic, coords, window, bin_factor)):
                win = enhance_window(win, noise, **extra)
                if win.min() == win.max():
                    coord = coords[index]
                    x, y = (coord.x, coord.y) if hasattr(coord, 'x') else (coord[1], coord[2])
                    _logger.warn("Window %d at coordinates %d,%d has an issue - clamp_window may need to be increased"%(index+1, x, y))
                if single_stack:
                    ndimage_file.write_image(output, win, len(global_selection), header=dict(apix=extra['apix']))
                    global_selection.append((len(global_selection)+1, fid, index+1, ))
                else:
                    ndimage_file.write_image(output, win, index, header=dict(apix=extra['apix']))
            #_logger.info("Extract %d windows from movie %d frame %d - %d of %d - finished"%(len(coords), fid, frame, i, tot))
    except ndimage_file.InvalidHeaderException:
        _logger.warn("Skipping: %s - invalid header"%filename)
        return filename, 0, os.getpid()
    if len(global_selection) > 0:
        format.write(output, numpy.asarray(global_selection), prefix="sel_", header="id,micrograph,stack_id".split(','))
    return filename, len(coords), os.getpid()

def iter_micrographs(filename, index=None, bin_factor=1.0, sigma=1.0, disable_bin=False, invert=False, window=None, gain=None, **extra):
    ''' Read a micrograph from a file and perform preprocessing
    
    :Parameters:
            
        filename : str
                   Filename for micrograph
        index : int
                Index of micrograph in stack
        bin_factor : float
                    Downsampling factor
        sigma : float
                Gaussian highpass filtering factor (sigma/window)
        disable_bin : bool    
                      If True, do not downsample the micrograph
        invert : bool
                 If True, invert the contrast of the micrograph (CCD Data)
        window : int
                 Size of the window in pixels
        gain : array
               Gain correction/normalization image
        extra : dict
                Unused extra keyword arguments
    
    :Returns:
            
        mic : array
              Micrograph image
    '''
    
    assert(window is not None)
    if isinstance(filename, tuple):
        filename = filename[1][index]
        index=None
    
    if ndimage_file.count_images(filename) == 1: index=None
    
    for mic in ndimage_file.iter_images(filename, index):
        _logger.debug("Read micrograph")
        mic = mic.astype(numpy.float32)
        if gain is not None and index is not None: mic *= gain
        if bin_factor > 1.0 and not disable_bin: 
            _logger.debug("Downsample by %f"%bin_factor)
            mic = ndimage_interpolate.downsample(mic, bin_factor)
        if invert:
            _logger.debug("Invert micrograph")
            ndimage_utility.invert(mic, mic)
        if sigma > 0.0:
            _logger.debug("Filter by %f"%(sigma/float(window)))
            mic = ndimage_filter.gaussian_highpass(mic, sigma/float(window), True)
        yield mic

def read_micrograph(filename, index=0, bin_factor=1.0, sigma=1.0, disable_bin=False, invert=False, window=None, gain=None, **extra):
    ''' Read a micrograph from a file and perform preprocessing
    
    :Parameters:
            
        filename : str
                   Filename for micrograph
        index : int
                Index of micrograph in stack
        bin_factor : float
                    Downsampling factor
        sigma : float
                Gaussian highpass filtering factor (sigma/window)
        disable_bin : bool    
                      If True, do not downsample the micrograph
        invert : bool
                 If True, invert the contrast of the micrograph (CCD Data)
        window : int
                 Size of the window in pixels
        gain : array
               Gain correction/normalization image
        extra : dict
                Unused extra keyword arguments
    
    :Returns:
            
        mic : array
              Micrograph image
    '''
    
    assert(window is not None)
    if isinstance(filename, tuple):
        filename = filename[1][index]
        index=None
    
    if ndimage_file.count_images(filename) == 1: index=None
    _logger.debug("Read micrograph")
    mic = ndimage_file.read_image(filename, index, **extra)
    
    if issubclass(numpy.dtype(mic.dtype).type, numpy.integer):
        _logger.warn("You are processing an image that is not gain corrected!")
    _logger.debug("Convert to 32 bit")
    mic = mic.astype(numpy.float32)
    if gain is not None and index is not None: mic *= gain
    if bin_factor > 1.0 and not disable_bin: 
        _logger.debug("Downsample by %f"%bin_factor)
        mic = ndimage_interpolate.downsample(mic, bin_factor)
    if invert:
        _logger.debug("Invert micrograph")
        ndimage_utility.invert(mic, mic)
    if sigma > 0.0:
        _logger.debug("Filter by %f"%(sigma/float(window)))
        mic = ndimage_filter.gaussian_highpass(mic, sigma/float(window), True)
    return mic
            
def generate_noise(filename, noise="", output="", noise_stack=True, window=None, pixel_diameter=None, **extra):
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
        window : int
                 Size of the window in pixels
        pixel_diameter : int
                         Diameter of particle in pixels
        extra : dict
                Unused extra keyword arguments
    
    :Returns:
            
        noise_win : array
                    Noise window
    '''
    
    if noise != "":  
        noise_img = ndimage_file.read_image(noise)
        if noise_img.shape[0] == window: return noise_img
        _logger.warn("Noise window size does not match window size: %d != %d - generating new noise window!"%(noise_img.shape[0], window))
    noise_file = format_utility.add_prefix(output, "noise_")
    if os.path.exists(noise_file):
        _logger.warn("Found cached noise file: %s - delete if you want to regenerate"%noise_file)
        noise_img = ndimage_file.read_image(noise_file)  
        if noise_img.shape[0] == window: return noise_img
        _logger.warn("Noise window size does not match window size: %d != %d - generating new noise window!"%(noise_img.shape[0], window))
    tot = ndimage_file.count_images(filename)
    if tot > 1:
        return None
    window = int(window)
    mic = read_micrograph(filename, window=window, **extra)
    
    template = numpy.zeros((window,window))
    template[1:window-1, 1:window-1]=1 #/(window-1)**2
    
    cc_map = ndimage_utility.cross_correlate(mic, template)
    numpy.fabs(cc_map, cc_map)
    cc_map -= float(numpy.max(cc_map))
    cc_map *= -1
    peak1 = ndimage_utility.find_peaks_fast(cc_map, window/2)
    peak1 = numpy.asarray(peak1).squeeze()
    index = numpy.argsort(peak1[:,0])[::-1]
    peak1 = peak1[index].copy().squeeze()
    
    best = (1e20, None)
    for i, win in enumerate(ndimage_utility.for_each_window(mic, peak1, window, 1.0)):
        win = enhance_window(win, None, None, **extra)
        std = win.std()
        if std < best[0]: best = (std, win)
        if noise_stack and i < 11: 
            ndimage_file.write_image(noise_file, win, i)
    noise_win = best[1]
    if not noise_stack:
        ndimage_file.write_image(noise_file, noise_win)
    return noise_win

def enhance_window(win, noise_win=None, norm_mask=None, mask=None, clamp_window=0.0, disable_enhance=False, disable_normalize=False, **extra):
    '''Enhance the window with a set of filtering and normalization routines
    
    :Parameters:
        
        win : array
              Input raw window
        noise_win : array
                    Noise window
        norm_mask : array
                    Disk mask with `radius` that keeps data outside the disk
        mask : array
               Disk mask with `radius` that keeps data inside the disk
        clamp_window : float
                       Number of standard deviations to replace extreme values using a Gaussian distribution
        disable_enhance : bool
                          Disable all enhancement
        disable_normalize : bool
                           Disable normalization
        extra : dict
                Unused extra keyword arguments
    
    :Returns:
            
        win : array
              Enhanced window
    '''
    
    win = win.astype(numpy.float32)
    if not disable_enhance:
        _logger.debug("Removing ramp from window")
        ndimage_filter.ramp(win, win)
        
    if clamp_window > 0: 
        _logger.debug("Removing outlier pixels")
        ndimage_utility.replace_outlier(win, clamp_window, out=win)
    if win.max() == win.min(): return win

    if noise_win is not None and not disable_enhance: 
        _logger.debug("Improving contrast with histogram fitting")
        ndimage_filter.histogram_match(win, mask, noise_win, out=win)
    if not disable_normalize and norm_mask is not None:
        try:
            win[:]=ndimage_utility.normalize_standard(win, norm_mask)#, out=win)
        except:
            _logger.error('Normalize: %f, %f'%(norm_mask.sum(), win.mean()))
            raise
    _logger.debug("Finished Enhancment")
    return win

def read_coordinates(coordinate_file, good_file="", spiderid=None, **extra):
    ''' Read a coordinate file (use `good_file` to select a subset if specified)
    
    :Parameters:
            
        coordinate_file : str
                          Filename for input coordinate file
        good_file : str
                    Filename for input selection file
        extra : dict
                Unused extra keyword arguments
    
    :Returns:
            
        coords : list
                 List of coordinates (namedtuple)
    '''
    
    try:
        select = format.read(good_file, numeric=True, spiderid=spiderid) if good_file != "" else None
    except:
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.exception("Skipping: %s"%good_file)
        else:
            _logger.warn("Cannot find selection file: %s"%good_file)
        raise
    try:
        blobs = format.read(coordinate_file, numeric=True, spiderid=spiderid)
    except:
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.exception("Skipping: %s"%coordinate_file)
        else:
            _logger.warn("Cannot find coordinate file: %s"%coordinate_file)
        raise
    if select is not None: 
        blobs = selection_utility.select_subset(blobs, select)
    return blobs


def init_root(files, param):
    # Initialize global parameters for the script
    
    if len(files) == 0: return []
    filename = files[0] if len(files) > 0 else param['finished'][0]
    if mpi_utility.is_root(**param):
        tot = ndimage_file.count_images(filename)
        if param['frame_align'] != "" and tot == 1:
            files = spider_utility.single_images(files)
            _logger.info("Processing example movie: %s"%str(filename))
        
        if tot > 0:
            if param['frame_align'] != "":
                _logger.info("Processing %d aligned movie micrographs"%len(files))
            else:
                _logger.info("Processing %d movie micrographs"%len(files))
        else:
            _logger.info("Processing %d micrographs"%len(files))
    
    files = mpi_utility.broadcast(files, **param)
    return files

def initialize(files, param):
    # Initialize global parameters for the script
    
    if len(files) == 0 and len(param['finished']) == 0: return []
    filename = files[0] if len(files) > 0 else param['finished'][0]
    param['global_selection']=[]
    
    if len(param['finished']) > 0 and param['single_stack']:
        raise ValueError, "Cannot restart with --single-stack -- please use --force or force: True or delete the output files"
    
    spider_params.read(param['param_file'], param)
    if param['mask_diameter'] <= 0.0:
        param['mask_diameter'] = param['pixel_diameter']
    else:
        param['mask_diameter']=int(param['mask_diameter']/param['apix'])
    if mpi_utility.is_root(**param):
        _logger.info("Processing %d files"%len(files))
        _logger.info("Particle diameter (in pixels): %d"%(param['pixel_diameter']))
        _logger.info("Mask diameter (in pixels): %d"%(int(param['mask_diameter'])))
        param['window']=int(param['window'])
        if not param['disable_even'] and (param['window']%2)==1:
            param['window'] += 1
            _logger.info("Window size: %d (Forced Even)"%(param['window']))
        else:
            _logger.info("Window size: %d"%(param['window']))
        if param['gain_file'] != "":
            _logger.info("Gain correct with %s"%param['gain_file'])
        if param['sigma'] > 0: _logger.info("High pass filter: %f"%(param['sigma'] / float(param['window'])))
        if param['bin_factor'] > 1 and not param['disable_bin']: _logger.info("Decimate micrograph by %d"%param['bin_factor'])
        if param['invert']: _logger.info("Inverting contrast of the micrograph")
        if not param['disable_enhance']:
            _logger.info("Enhancement to windows:")
            _logger.info("- Ramp filter")
            _logger.info("- Dedust: %f"%param['clamp_window'])
            _logger.info("- Histogram matching")
            if not param['disable_normalize']: _logger.info("- Normalize (xmipp style)")
        param['count'] = numpy.zeros(2, dtype=numpy.int)
        if len(files) > 0 and not isinstance(filename, tuple):
            _logger.info("Extracting windows from micrograph stacks of movie frames")
    
    if mpi_utility.is_root(**param):
        selection_file = format_utility.parse_header(param['selection_file'])[0]
        if os.path.exists(selection_file):
            select = format.read(param['selection_file'], numeric=True)
            file_count = len(files)
            if len(select) > 0:
                files=selection_utility.select_file_subset(files, select)
                finished=list(param['finished'])
                del param['finished'][:]
                param['finished'].extend( selection_utility.select_file_subset(finished, select) )
            _logger.info("Assuming %s is a micrograph selection file - found %d micrographs of %d"%(selection_file, len(files), file_count))
        
        if isinstance(filename, tuple):
            tot = len(filename[1])
        else:
            tot = ndimage_file.count_images(filename)
        for filename in param['finished']:
            if not ndimage_file.valid_image(filename):
                files.append(filename)
            elif tot > 1:
                id = filename[0] if isinstance(filename, tuple) else filename
                coord = read_coordinates(spiderid=id, **param)
                align = format.read(param['frame_align'], spiderid=id, numeric=True, id_len=param['id_len'])
                ncoord=len(coord)
                # Todo only add frames that require processing
                param['output']=strip_frame_tag(param['output'])
                for i in xrange(tot):
                    frame = spider_utility.spider_id(filename[1][i]) if isinstance(filename, tuple) else align[i].id+1
                    output = format_utility.add_prefix(param['output'], 'frame_%d_'%(frame))
                    frame_stack = spider_utility.spider_filename(output, id, param['id_len'])
                    if not os.path.exists(frame_stack):
                        _logger.info("Found incomplete frame crop: %s"%(frame_stack))
                        files.append(filename)
                        break
                    nimage = ndimage_file.count_images(frame_stack)
                    if nimage != ncoord:
                        _logger.info("Found partial stack: %d != %d"%(ncoord, nimage))
                        files.append(filename)
                        break
            else:
                id = filename[0] if isinstance(filename, tuple) else filename
                ncoord = len(read_coordinates(spiderid=id, **param))
                nimage = ndimage_file.count_images(spider_utility.spider_filename(param['output'], id, param['id_len']))
                if nimage != ncoord:
                    _logger.info("Found partial stack: %d != %d"%(ncoord, nimage))
                    files.append(filename)
               
    files = mpi_utility.broadcast(files, **param)
    
    if len(files) == 0:
        param['noise']=None
    else:
        if not isinstance(filename, tuple):
            if mpi_utility.is_root(**param):
                param['noise'] = generate_noise(filename, **param)
            mpi_utility.barrier(**param)
            if mpi_utility.is_client_strict(**param):
                param['noise'] = generate_noise(filename, **param)
        else: param['noise']=None
    if param['gain_file'] != "":
        param['gain'] =  ndimage_file.read_image(param['gain_file'])
    
    
    param.update(ndimage_file.cache_data())
    assert(int(param['mask_diameter']) < int(param['window']))
    param['mask'] = ndimage_utility.model_disk(int(param['mask_diameter']/2.0), (param['window'], param['window']))
    param['norm_mask']=param['mask']*-1+1
    assert(param['mask'].sum()>0)
    assert(param['mask'].sum()<numpy.prod(param['mask'].shape))
    assert(param['norm_mask'].sum()>0)
    return files

def strip_frame_tag(filename):
    ''' Strip the frame tag from an output window stack filename
    
    :Parameters:
    
        filename : str
                   Filename with frame_%d_
    
    :Returns:
    
        filename : str
                   Filename with frame_%d_ stripped off
    '''
    
    base = os.path.basename(filename)
    if base.startswith('frame'):
        root = os.path.dirname(filename)
        n = base.find('_')
        n = base.find('_', n+1)
        base = base[n+1:]
        filename = os.path.join(root, base)
    return filename

def reduce_all(val, count, id_len, **extra):
    # Process each input file in the main thread (for multi-threaded code)

    filename, total, pid = val
    count[0]+= total
    count[1]+= 1
    if isinstance(filename, tuple): filename=filename[0]
    return filename, str(filename)+" - %d windows - %d windows in total in %d files - %1f gigs"%(total, count[0], count[1], psutil.Process(pid).get_memory_info().rss/131072.0)

def finalize(files, count=[0], **extra):
    # Finalize global parameters for the script
    _logger.info("Extracted %d windows"%count[0])
    _logger.info("Completed")
    
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
    
    group = OptionGroup(parser, "Cropping", "Options to crop particles from micrographs", id=__name__) #, gui=dict(root=True, stacked="prepComboBox"))
    
    group.add_option("", single_stack=False,       help="Crop all files into a single stack")
    group.add_option("", invert=False,             help="Invert the contrast on the micrograph (usually for raw CCD micrographs)")
    group.add_option("", noise="",                 help="Use specified noise file rather then automatically generate one", gui=dict(filetype="open"))
    group.add_option("", gain_file="",             help="Perform gain correction with given norm image", gui=dict(filetype="open"))
    
    egroup = OptionGroup(parser, "Enhancement", "Enhancement for the windows")
    egroup.add_option("", disable_even=False,       help="Disable forcing windows to be even")
    egroup.add_option("", disable_enhance=False,    help="Disable window post-processing: ramp, contrast enhancement")
    egroup.add_option("", disable_bin=False,        help="Disable micrograph decimation")
    egroup.add_option("", disable_normalize=False,  help="Disable XMIPP normalization")
    egroup.add_option("", clamp_window=2.5,         help="Number of standard deviations to replace extreme values using a Gaussian distribution (0 to disable)")
    egroup.add_option("", sigma=1.0,                help="Highpass factor: 1 or 2 where 1/window size or 2/window size (0 to disable)")
    egroup.add_option("", mask_diameter=0.0,        help="Mask multiplier for Relion (in Angstroms) - 0 mean uses particle_diameter")
    group.add_option_group(egroup)
    
    mgroup = OptionGroup(parser, "Movies", "Crop frames from movie micrographs")
    mgroup.add_option("", frame_align="",           help="Translational alignment parameters for individual frames")
    mgroup.add_option("", frame_beg=0,              help="Index of first frame")
    mgroup.add_option("", frame_end=-1,             help="Index of last frame")
    group.add_option_group(mgroup)
    
    #group.add_option("-r", pixel_radius=0,         help="Radius of the expected particle (if default value 0, then overridden by SPIDER params file, `param-file`)")
    #group.add_option("",   window=1.0,             help="Size of the output window or multiplicative factor if less than particle diameter (overridden by SPIDER params file, `param-file`)")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", "--micrograph-files", input_files=[],         help="List of input filenames containing micrographs, e.g. mic_*.mrc ", required_file=True, gui=dict(filetype="open"), regexp=spider_utility.spider_searchpath)
        pgroup.add_option("-o", "--particle-file",   output="",               help="Output filename for window stack with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-l", coordinate_file="",                           help="Input filename template containing particle coordinates with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="open"), required_file=True)
        pgroup.add_option("-s", selection_file="",                            help="Selection file for a subset of micrographs", gui=dict(filetype="open"), required_file=False)
        pgroup.add_option("-g", good_file="",                                 help="Selection file template for subset of good particles", gui=dict(filetype="open"), required_file=False)
        parser.change_default(log_level=3)
        spider_params.setup_options(parser, pgroup, True)

def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    
    if len(options.input_files)>0:
        n = ndimage_file.count_images(options.input_files[0])
        if n > 1:
            frame_end = n if options.frame_end < 0 else options.frame_end
            options.output = format_utility.add_prefix(options.output, 'frame_%d_'%(frame_end))
    
    if options.bin_factor == 0.0: raise OptionValueError, "Bin factor cannot be zero (--bin-factor)"
    if options.single_stack:
        if options.use_MPI: raise OptionValueError, "You cannot use --single-stack with MPI"
        if options.worker_count > 1:
            options.worker_count = 1
            _logger.warn("Setting --worker-count to 1 - This is necessary when using --single-stack mode")


def flags():
    ''' Get flags the define the supported features
    
    :Returns:
        
        flags : dict
                Supported features
    '''
    
    return dict(description = '''Crop windows containing particles from a micrograph
                         
                        Example: Run from the command line on a single node
                        
                        $ %prog micrograph_01.spi -o particle-stack_01.spi -p params.spi -s sndc_01.spi
                      ''',
                supports_MPI=True, 
                supports_OMP=True,
                use_version=True)

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__)

def dependents(): return []

if __name__ == "__main__": main()


