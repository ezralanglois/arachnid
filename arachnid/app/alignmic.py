''' Micrograph alignment for tilt, defocus, or dose fractionated series

.. Created on Jan 24, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import matplotlib
matplotlib.use('Agg')
import pylab
from ..core.app import program
#from ..core.util import plotting
from ..core.metadata import spider_utility, spider_params, format_utility, format #, format_utility
from ..core.image import ndimage_file, ndimage_utility, eman2_utility, align, rotate #, ctf #, analysis
from ..core.image.formats import mrc as mrc_file
from ..core.parallel import mpi_utility
import numpy, logging, scipy, scipy.stats #, os, scipy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, id_len=0, disable_align=False, reverse=False, recalc_avg=False, **extra):
    ''' Esimate the defocus of the given micrograph
    
    :Parameters:
    
    filename : str
               Input micrograph, stack or power spectra file
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    filename : str
               Current filename
    '''
    
    if isinstance(filename, tuple):
        id = filename[0]
    else:
        id = spider_utility.spider_id(filename, id_len)
    
    spider_utility.update_spider_files(extra, id, 'output')
    output=extra['output']
    
    if isinstance(filename, tuple):
        filename=filename[1]
        if reverse: filename.reverse()
        # reverse order
    else:
        try:
            tot = mrc_file.count_images(filename)
        except: tot = ndimage_file.count_images(filename)
        index = numpy.arange(tot, dtype=numpy.int)
        if reverse: index = index[::-1]
        filename = [(filename, i) for i in index]
        #numpy.random.shuffle(index)
    param=numpy.zeros((len(filename), 2))
    
    missing = 0
    if not recalc_avg:
        if not disable_align:
            param, missing = align_micrographs(filename, id, param, **extra)
            format.write(output, param[:, :2], prefix="trans_", header="dx,dy".split(','), default_format=format.spiderdoc)
    else:
        param = format.read(output, prefix="trans_", ndarray=True)[0][:, 1:]
    
    _logger.info("Averging aligned micrographs")
    if disable_align: param= None
    sum = average(filename, param, extra['experimental'])
    _logger.info("Averging aligned micrographs - finished")
    ndimage_file.write_image(output, sum)
    return filename, missing

def align_micrographs(files, id, param, output, quality_test=False, frame_limit=0, **extra):
    '''
    '''
    
    sum = read_micrograph(files[0], **extra)
    ref = sum.copy()
    #prev = sum.copy()
    pow_output = format_utility.add_prefix(output, "pow_")
    powi_output = format_utility.add_prefix(output, "pow_interp_")
    powo_output = format_utility.add_prefix(output, "pow_orig_")
    cc_output = format_utility.add_prefix(output, "cc_")
    missing=0
    if frame_limit == 0 or frame_limit > len(files): frame_limit = len(files)
    x = numpy.arange(len(param))
    
    if extra['experimental']:
        if 1 == 0:
            data = numpy.zeros((frame_limit, 512*512))
            for i, f in enumerate(files[1:frame_limit]):
                data[i, :] = read_micrograph(f, **extra)[:512, :512].ravel()
            d, V = numpy.linalg.eig(numpy.cov(data-data.mean()))
            idx = numpy.argsort(d)[::-1]
            V=V[idx]
            try:
                pylab.clf()
                pylab.scatter(V[:, 0], V[:, 1], marker='o', c='r')
                
                for label, x, y in zip(idx+1, V[:, 0], V[:, 1]):
                    pylab.annotate(
                        str(label), 
                        xy = (x, y), xytext = (-20, 20),
                        textcoords = 'offset points', ha = 'right', va = 'bottom',
                        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
                
                pylab.savefig(format_utility.new_filename(output, 'eig01_', ext="png"), dpi=200)
                pylab.clf()
                pylab.scatter(V[:, 1], V[:, 2], marker='o', c='r')
                
                for label, x, y in zip(idx+1, V[:, 1], V[:, 2]):
                    pylab.annotate(
                        str(label), 
                        xy = (x, y), xytext = (-20, 20),
                        textcoords = 'offset points', ha = 'right', va = 'bottom',
                        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
                
                pylab.savefig(format_utility.new_filename(output, 'eig12_', ext="png"), dpi=200)
                pylab.clf()
                pylab.scatter(V[:, 2], V[:, 3], marker='o', c='r')
                
                for label, x, y in zip(idx+1, V[:, 2], V[:, 3]):
                    pylab.annotate(
                        str(label), 
                        xy = (x, y), xytext = (-20, 20),
                        textcoords = 'offset points', ha = 'right', va = 'bottom',
                        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
                
                pylab.savefig(format_utility.new_filename(output, 'eig23_', ext="png"), dpi=200)
            except: _logger.exception("bug")
        
    
    if 1 == 1:
        last = read_micrograph(files[frame_limit-1], **extra)
        dxl, dyl, res = align_pair(last, ref, frame_limit-1, cc_output, quality_test, **extra)
        _logger.info("Target: %d/%d - Shift: %d, %d"%(id, frame_limit-1, dxl, dyl))
        
        fitx = numpy.poly1d(numpy.polyfit([0, frame_limit],[0, dxl],1))
        fity = numpy.poly1d(numpy.polyfit([0, frame_limit],[0, dyl],1))
        sum2=sum.copy()
        sum3=sum.copy()
        for i, f in enumerate(files[1:frame_limit]):
            _logger.info("Processing: %d of %d"%(i, frame_limit))
            img = read_micrograph(f, **extra)
            
            sum3+=img
            pow = ndimage_utility.perdiogram(eman2_utility.decimate(sum3, 2))
            ndimage_file.write_image(powo_output, pow, i)
            
            sum2+=eman2_utility.fshift(img, fitx(i), fity(i))
            pow = ndimage_utility.perdiogram(eman2_utility.decimate(sum2, 2))
            ndimage_file.write_image(powi_output, pow, i)
            
            
    
    for i, f in enumerate(files[1:frame_limit]):
        try:
            img = read_micrograph(f, **extra)
        except:
            missing+=1
            continue
        numpy.divide(sum, i+1, ref)
        dx, dy, res = align_pair(img, ref, id, cc_output, quality_test, **extra)
        #dx2, dy2, res2 = align_pair(img, prev, id, cc_output, quality_test, **extra)
        param[i+1, :]=(dx, dy)
        try:
            pylab.clf()
            pylab.plot(x, param[:, 0], 'r-.')
            pylab.plot(x, param[:, 1], 'b--')
            pylab.savefig(format_utility.new_filename(output, 'tracking_iter', ext="png"), dpi=200)
        except: pass
        simg = eman2_utility.fshift(img, dx, dy)
        '''
        p1 = snr(prev.ravel(), img.ravel())
        p2 = snr(prev.ravel(), simg.ravel())
        p3 = snr(ref.ravel(), img.ravel())
        p4 = snr(ref.ravel(), simg.ravel())
        '''
        sum+=simg
        _logger.info("%d/%d - Shift: %d, %d -- %f"%(id, i+1, dx, dy, res))#, p1, p2, p3, p4)) -- sing: %f, %f | avg: %f, %f
        #_logger.info("%d/%d - Shift: %d, %d -- %f - Shift-prev: %d, %d -- %f"%(id, i+1, dx, dy, res, dx2, dy2, res2))#, p1, p2, p3, p4)) -- sing: %f, %f | avg: %f, %f
        if quality_test:
            pow = ndimage_utility.perdiogram(eman2_utility.decimate(sum, 2))
            ndimage_file.write_image(pow_output, pow, i)
        #prev = simg
    
    
    
    try:
        pylab.clf()
        fit = numpy.poly1d(numpy.polyfit(x,param[:, 0],1))
        pylab.plot(x, param[:, 0], 'r+', x, fit(x), '--r', label='x-direction')
        fit = numpy.poly1d(numpy.polyfit(x,param[:, 1],1))
        pylab.plot(x, param[:, 1], 'bo', alpha=0.5)
        pylab.plot(x, fit(x), '--b', label='y-direction')
        pylab.legend()
        pylab.savefig(format_utility.new_filename(output, 'drift_fit', ext="png"), dpi=200)
    except: pass
    
    try:
        pylab.clf()
        pylab.scatter(param[:, 0], param[:, 1], marker='o', c='r')
        for label, x, y in zip(numpy.arange(1,1+len(param)), param[:, 0], param[:, 1]):
            pylab.annotate(
                str(label), 
                xy = (x, y), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        pylab.savefig(format_utility.new_filename(output, 'drift_dir', ext="png"), dpi=200)
    except: _logger.exception("bug")
    
    param *= extra['bin_factor']
    return param, missing

def snr(img1, img2):
    '''
    '''
    
    p = scipy.stats.pearsonr(img1.ravel(), img2.ravel())[0]
    return p/(1.0-p)

def align_win(img, ref, id, cc_output, quality_test, apix, resolution, pixel_diameter, offset=0, **extra):
    '''
    '''
    
    ref = eman2_utility.gaussian_low_pass(ref, apix/resolution)
    #ref = eman2_utility.gaussian_high_pass(ref, 0.5/pixel_diameter)
    #ref = ndimage_utility.dog(ref, pixel_diameter/8.0)
    #img = ndimage_utility.dog(img, pixel_diameter/8.0)
    window_size = (numpy.min(ref.shape)-offset*2)/2
    xc, yc = offset+window_size/2, offset+window_size/2
    if 1 == 1:
        img1 = ndimage_utility.crop_window(img, xc, yc, window_size/2)
        ref1 = ndimage_utility.crop_window(ref, xc, yc, window_size/2)
    else:
        img1=img
        ref1=ref
    if 1 == 0:
        mask = numpy.ones((img1.shape[0]/2, img1.shape[1]/2))
        cc_map = ndimage_utility.cross_correlate(img1, ref1, False)
        cc_map /= (ndimage_utility.local_variance(img1, mask)+1e-5)
        x, y = select_peak(cc_map)
    elif 1 == 0:
        best = (-1e20, 0, 0)
        for x in xrange(-2, 3):
            for y in xrange(-2, 3):
                simg = eman2_utility.fshift(img1, x, y)
                p = snr(ref1.ravel(), simg.ravel())
                if p > best[0]: best=(p, x, y)
        x, y = best[1:]
    elif 1 == 0:
        y, x, z = align.align_translation(img1, ref1, (10, 10))[:3]
        assert(z==0)
    else:
        cc_map = ndimage_utility.cross_correlate(img1, ref1, False)
        x, y = select_peak(cc_map)
    return x, y, 0

def align_win2(img, ref, id, cc_output, quality_test, apix, resolution, pixel_diameter, offset=0, **extra):
    '''
    '''
    
    ref = eman2_utility.gaussian_low_pass(ref, apix/resolution)
    window_size = (numpy.min(ref.shape)-offset*2)/2
    #cc_map_sum = None
    xc, yc = offset, offset
    for i in xrange(2):
        for j in xrange(2):
            img1 = ndimage_utility.crop_window(img, xc, yc, window_size/2)
            ref1 = ndimage_utility.crop_window(ref, xc, yc, window_size/2)
            cc_map = ndimage_utility.cross_correlate(img1, ref1)
            x, y = select_peak(cc_map)
            _logger.info("%d: %f - %d, %d"%(id, cc_map[x,y], x, y))
            yc = ref.shape[1]-window_size-offset
        xc = ref.shape[0]-window_size-offset
    return x, y, 0

def align_pair(img, ref, id, cc_output, quality_test, apix, resolution=20.0, experimental=False, **extra):
    '''
    '''
    
    if experimental: return align_win(img, ref, id, cc_output, quality_test, apix, resolution, **extra)
    
    if not quality_test:
        ref = eman2_utility.gaussian_low_pass(ref, apix/resolution)
    cc_map = ndimage_utility.cross_correlate(img, ref)
    
    if quality_test:
        best = (-1e20, None)
        for i, f in enumerate(scipy.linspace(apix*2.25, 30, 10)):
            fcc_map = eman2_utility.gaussian_low_pass(cc_map, apix/f)
            ndimage_file.write_image(cc_output, fcc_map, i)
            x, y = select_peak(fcc_map)
            d=fcc_map[x,y]
            _logger.info("%d. %f: %f - %d, %d"%(id, f, d, x, y))
            if d > best[0]: 
                best = (d, (x, y, f))
        return best[1]
    else:
        return select_peak(cc_map)+(0, )

def select_peak(cc_map):
    '''
    '''
    
    y,x = numpy.unravel_index(numpy.argmax(cc_map), cc_map.shape)
    x -= cc_map.shape[1]/2
    y -= cc_map.shape[0]/2
    return -x, -y
    
def average(filename, param, experimental):
    '''
    '''
    
    sum = read_image(filename[0])
    for i, f in enumerate(filename[1:]):
        _logger.info("Average frame: %d of %d"%(i+1, len(filename)))
        img = read_image(f)
        if experimental and 1 == 0:
            sum+=rotate.rotate_image(img, 0, param[i+1, 0], param[i+1, 1])
        elif param is not None:
            sum+=eman2_utility.fshift(img, param[i+1, 0], param[i+1, 1])
        else: sum += img
    return sum

def read_image(filename):
    if isinstance(filename, tuple): filename, index = filename
    else: index=None
    if 1 == 1:
        return ndimage_file.read_image(filename, index)
    try:
        mic = mrc_file.read_image(filename, index)
    except:
        mic = ndimage_file.read_image(filename, index)
    mic = mic.astype(numpy.float)
    return mic
    
def read_micrograph(filename, bin_factor, invert, pixel_diameter, **extra):
    ''' Read an process a micrograph
    '''
    
    mic = read_image(filename)
    #mic = eman2_utility.gaussian_high_pass(mic, 0.5/(pixel_diameter*bin_factor))
    if bin_factor > 1: mic = eman2_utility.decimate(mic, bin_factor)
    if invert: ndimage_utility.invert(mic, mic)
    return mic
    
    
def init_root(files, param):
    # Initialize global parameters for the script
    
    if mpi_utility.is_root(**param):
        _logger.info("Processing %d micrographs"%len(files))
        _logger.info("Align: %d"%(not param['disable_align']))
        if not param['disable_align']:
            _logger.info("Bin-factor: %f"%param['bin_factor'])
            _logger.info("Invert: %d"%param['invert'])
            _logger.info("Pixel Diameter: %d"%(param['pixel_diameter']))
            _logger.info("Experimental: %d"%(param['experimental']))
            _logger.info("Quality test: %d"%(param['quality_test']))
        param['missing_frames']=[0]
        
        try: tot = mrc_file.count_images(files[0])
        except: tot = ndimage_file.count_images(files[0])
        if tot == 1:
            _logger.info("Reorganizing individual files")
            files = spider_utility.single_images(files)
            for f in files:
                _logger.info("Reorganizing individual files: %d"%f[0])
                if f[0] == 0:
                    raise ValueError, "Cannot have 0 ID"
        else:
            _logger.info("Frames in a stack")
    files=mpi_utility.broadcast(files, **param)
    return sorted(files)

def reduce_all(filename, missing_frames, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    filename, missing = filename
    missing_frames[0]+=missing
    return filename

def finalize(files, missing_frames, **extra):
    # Finalize global parameters for the script
    
    _logger.info("Number of missing frames: %d"%missing_frames[0])
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import OptionGroup 
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input filenames containing micrographs, window stacks or power spectra", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for defocus file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-s", select="",      help="Selection file")
        pgroup.add_option("",   dpi=72,         help="Resolution in dots per inche for color figure images")
        pgroup.add_option("",   max_range=0,   help="Maximum translation range")
        pgroup.add_option("",   frame_limit=0,   help="Maximum translation range")
        pgroup.add_option("",   window_size=0,  help="Use only a window to translate")
        pgroup.add_option("",   experimental=False,  help="Use experimental alignment")
        #pgroup.add_option("",   disk_mult=0.0,  help="Use experimental alignment with cc")
        pgroup.add_option("",   quality_test=False, help="Write out progressive power spectra and cc filtered maps")
        pgroup.add_option("",   reverse=False,    help="Reverse the alignment order")
        pgroup.add_option("",   resolution=20.0,    help="Filter for micrograph alignment")
        pgroup.add_option("",   recalc_avg=False,    help="Recalculate average from known translation")
        
        
        parser.change_default(log_level=3)
    
    group = OptionGroup(parser, "Alignment", "Options to control alignment",  id=__name__)
    group.add_option("", invert=False, help="Invert the contrast - used for unprocessed CCD micrographs")    
    group.add_option("", disable_align=False, help="Disable micrograph alignment")
    pgroup.add_option_group(group)
    
def setup_main_options(parser, group):
    # Setup options for the main program
    
    parser.change_default(bin_factor=1)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if not options.disable_align:
        if options.param_file == "": raise OptionValueError('SPIDER Params file empty')

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Align a series of micrographs
                        
                        http://
                        
                        $ ara-alignmic mic_*.ter -p params.ter -o align.ter
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=True,
        use_version = True,
    )

def dependents(): return [spider_params]
if __name__ == "__main__": main()


