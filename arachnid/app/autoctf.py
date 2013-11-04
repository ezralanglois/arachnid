''' Automatically determine defocus and select good power spectra

.. Created on Jan 11, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.util import plotting
from ..core.metadata import spider_utility, format, format_utility, spider_params, selection_utility
from ..core.image import ndimage_file, ndimage_utility, ctf, ndimage_interpolate #, analysis
from ..core.parallel import mpi_utility
import os, numpy, logging #, scipy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, pow_color, bg_window, id_len=0, trunc_1D=False, experimental=False, **extra):
    ''' Esimate the defocus of the given micrograph
    
    :Parameters:
    
    filename : str
               Input micrograph, stack or power spectra file
    output : str
             Output defocus file
    bg_window
    id_len : int
             Max length of SPIDER ID
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    filename : str
               Current filename
    '''
    
    id = spider_utility.spider_id(filename, id_len)
    output = spider_utility.spider_filename(output, id)
    try:
        pow = generate_powerspectra(filename, **extra)
    except:
        _logger.warn("Skipping %s - unable to generate power spectra"%filename)
        _logger.exception("Unable to generate power spectra")
        vals = [id, 0.0, 0.0, 0.0, 0.0]
        return filename, vals
    
    if 1 == 0:
        ctf.smooth_2d(pow, extra['apix']/25.0)
        assert(numpy.alltrue(numpy.isfinite(pow)))
    
    
    raw = ndimage_utility.mean_azimuthal(pow)[:pow.shape[0]/2]
    raw[1:] = raw[:len(raw)-1]
    raw[:2]=0
    
    #raw = ndimage_utility.mean_azimuthal(pow)[1:pow.shape[0]/2+1]
    if 1 == 1:
        roo = ctf.subtract_background(raw, bg_window)
    else: roo = raw
    #raw = raw[:len(roo)]
    extra['ctf_output']=spider_utility.spider_filename(extra['ctf_output'], id)
    
    if 1 == 0:
        
        mask_radius=int(ctf.resolution(extra['mask_radius'], len(roo), **extra)) if extra['mask_radius']>0 else 10
        pol = ndimage_utility.polar(pow, rng=(mask_radius, pow.shape[0]/2-1)).copy()
        ndimage_file.write_image('test_polar.spi', pol)
        rad=32
        polavg = numpy.zeros((pol.shape[0]-rad, pol.shape[1]))
        beg=0
        for i in xrange(polavg.shape[0]):
            end = beg + rad
            polavg[i, :] = pol[beg:end].mean(0)
        
        ndimage_file.write_image('test_polar_mov_avg.spi', polavg)
    
    # initial estimate with outlier removal?
    beg = ctf.first_zero(roo, **extra)
    st = int(ctf.resolution(30.0, len(raw), **extra))
    if 1 == 0:
        plotting.plot_line('line.png', [ctf.resolution(i, len(raw), **extra) for i in xrange(st, len(raw))], raw[st:], dpi=300)
    
    # Estimate by drop in error
    end = ctf.energy_cutoff(numpy.abs(roo[beg:]))+beg
    if end == 0: end = len(roo)
    end2 = int(ctf.resolution(extra['rmax'], len(roo), **extra))
    if end > end2: end = end2
    
    res = ctf.resolution(end, len(roo), **extra)
    assert(numpy.alltrue(numpy.isfinite(pow)))
    cir = 0.0#ctf.estimate_circularity(pow, beg, end)
    freq = numpy.arange(len(roo), dtype=numpy.float)
    #sp-project /home/ryans/data/data_sept10/counted/*.spi --voltage 300 --apix 1.5844 --cs 2.26
    try:
        roo1, beg1 = ctf.factor_correction(roo, beg, end) if 1 == 1 else roo[beg:end]
    except:
        _logger.error("Unable to correct 1D for %d"%id)
        return filename, [id, 0, 0, 0, 0]
    roo1=roo1[beg1:]
    old=beg
    beg = beg1+beg
    # correct for noice and bfactor
    #defocus, err = ctf.estimate_defocus_spi(roo[beg:end], freq[beg:end], len(roo), **extra)[:2]
    if extra['cs'] == 0.0:
        defocus, err = ctf.estimate_defocus_orig(roo1, freq[beg:end], len(roo), **extra)[:2]
    else:
        defocus, err = ctf.estimate_defocus_spi(roo1, freq[beg:end], len(roo), **extra)[:2]
    
    if 1 == 0:
        #astigmatism(pow, beg, end)
        model = ctf.ctf_model_spi(defocus, freq, len(roo), **extra)**2
        avg = ctf.model_1D_to_2D(model)
        ndimage_file.write_image('test_2d_model.spi', avg)
        pol = ndimage_utility.polar(avg, rng=(mask_radius, pow.shape[0]/2-1)).copy()
        ndimage_file.write_image('test_2d_model_polar.spi', pol)
        hybrid = hybrid_model_data(pow* (ndimage_utility.model_disk(mask_radius, pow.shape)*-1+1), model, beg)
        ndimage_file.write_image('test_hybrid.spi', hybrid)
        
    
    vals = [id, defocus, err, cir, res]
    
    vextra=[]
    try:
        vextra.append(extra['defocus_val'][id].defocus)
    except:vextra.append(-1)
    
    if len(roo1) == 0: roo1 = roo[beg:end]
    if trunc_1D:
        roo2 = roo1
        rng = (beg, end)
    else:
        #roo2 = roo[beg:]
        if experimental:
            bg = ctf.estimate_background(raw, defocus, freq, len(roo), **extra)[0]
            _logger.info("found background: %d"%bg)
            roo = ctf.subtract_background(raw, bg)
        roo2, beg1 = ctf.factor_correction(roo, old, len(roo))
        rng = (old, len(roo))
        #roo2 = ctf.background_correct(raw, **extra)[old:]
    if experimental:
        model = ctf.ctf_model_spi(defocus, freq, len(roo), **extra)**2
        hpow = hybrid_model_data(pow, model, beg)
    else:
        hpow = pow
    if 1 == 0:
        img = color_powerspectra(hpow, roo, roo2, freq, "%d - %.2f - %.2f - %.2f - %.2f - %.2f"%tuple(vals+vextra), defocus, rng, raw=raw[:len(roo)], **extra)
    else:
        img = color_powerspectra(hpow, roo, roo2, freq, "%d - %.2f - %.2f - %.2f - %.2f - %.2f"%tuple(vals+vextra), defocus, rng, **extra)
    pow_color = spider_utility.spider_filename(pow_color, id)
    if 1 ==0:
        pow_color2 = format_utility.add_prefix(pow_color, 'avg_')
        ndimage_file.write_image(pow_color2, ndimage_utility.rotavg(pow))
    if hasattr(img, 'ndim'):
        ndimage_file.write_image(pow_color, img)
    else:
        img.savefig(os.path.splitext(pow_color)[0]+".png", dpi=extra['dpi'], bbox_inches='tight', pad_inches = 0.0)
    
    write_ctf(raw, freq, roo, bg_window, **extra)
    '''
    ;dat/dat   14-JAN-2013 AT 17:42:05   local/ctf/ctf_09755.dat
 ; Spa. freq.(1/A), Back. noise, Back. subtr. PS,  Env(f)**2
    1 4   0.00000       293.113       231.443       354.246     
    2 4  3.487723E-04   293.765       230.791       300.414     
    3 4  6.975447E-04   294.401       141.201       254.358     
    4 4  1.046317E-03   295.018       214.941       214.941     
    5 4  1.395089E-03   295.618       148.703       181.231
    ''' 
    
    return filename, vals

def astigmatism(pow, beg, end):
    '''
    '''
    
    # 1d defocus
    # Ellipse rings in yellow model
    
    pow = pow.copy()
    
    from arachnid.core.image import ndimage_filter
    from skimage.filter import tv_denoise
    import skimage.transform
    import scipy.ndimage
    pow -= pow.min()
    pow = numpy.log(pow+1)
    pow = scipy.ndimage.filters.median_filter(pow, 3)
    pow = ndimage_utility.replace_outlier(pow, 2.5)
    pow = ndimage_filter.filter_gaussian_highpass(pow, 0.5/(beg+2))
    pow = tv_denoise(pow, weight=0.5, eps=2.e-4, n_iter_max=200)
    #zero = (model.max()-model.min())/3 
    #minima=ctf.find_extrema(model, model<zero, numpy.argmin)
    #minval = numpy.min(numpy.diff(minima))
    #out = ndimage_filter.filter_gaussian_lowpass(out, 0.5/(minval-2)))
    pow *= ndimage_utility.model_disk(end, pow.shape)
    pow = ndimage_utility.histeq(pow)
    ndimage_file.write_image("astig_pow.spi", pow)
    mask=pow>numpy.histogram(pow, bins=256)[1][128]
    ndimage_file.write_image("astig_mask.spi", mask)
    mask, n = scipy.ndimage.measurements.label(mask)
    print "mask: ", n
    labels = numpy.arange(1, n+1)
    labels=labels[numpy.argsort([numpy.sum(mask==l) for l in labels])][::-1]
    
    
    for i in xrange(min(3,n)):
        sel = labels[i]==mask
        ndimage_file.write_image("astig_pow%d.spi"%(i+1), sel)
        print skimage.transform.hough_ellipse(sel)
        estimate_astigmatism(numpy.vstack(numpy.unravel_index(numpy.argwhere(labels[i]==mask), mask.shape)[::-1]).T)
    
    
def estimate_astigmatism(data):
    '''
    '''
    
    print 'data:', data.shape
    scatter = numpy.dot(data, data.T)
    eigenvalues, transform = numpy.linalg.eig(scatter)
    print 'Rotation matrix', transform
    
    # Step 3: Rotate back the data and compute radii.
    # You can also get the radii from smaller to bigger
    # with 2 / numpy.sqrt(eigenvalues)
    rotated = numpy.dot(numpy.linalg.inv(transform), data)
    print 'Radii', 2 * rotated.std(axis=1)
    

def hybrid_model_data(pow, model, beg, out=None):
    '''
    '''
    #import scipy.ndimage
    
    if out is None: out = pow.copy()
    else: out[:]=pow
    
    
    #from arachnid.core.image import ndimage_filter
    #out -= out.min()
    #out = numpy.log(out+1)
    #out = scipy.ndimage.filters.median_filter(out, 3)
    #out = ndimage_utility.replace_outlier(out, 2.5)
    #out = ndimage_filter.filter_gaussian_highpass(out, 0.5/(beg+2))
    #from skimage.filter import tv_denoise #0.1
    #out = tv_denoise(out, weight=0.5, eps=2.e-4, n_iter_max=200)
    #zero = (model.max()-model.min())/3 
    #minima=ctf.find_extrema(model, model<zero, numpy.argmin)
    #minval = numpy.min(numpy.diff(minima))
    #out = ndimage_filter.filter_gaussian_lowpass(out, 0.5/(minval-2)))
    
    #roo = ndimage_utility.mean_azimuthal(pow)[:pow.shape[0]/2]
    #roo[1:] = roo[:len(roo)-1]
    #roo[:2]=0
    
    #mask = ctf.model_1D_to_2D(model)
    #mask1=mask>numpy.histogram(mask, bins=512)[1][1]
    
    model = ctf.model_1D_to_2D(model)
    out = ndimage_utility.histeq(out)
    model = ndimage_utility.histeq(model)

    #normalize(out, model, mask1)
    #normalize(out, model, numpy.logical_not(mask1))
    
        
    
    n = pow.shape[0]/2
    out[:n, :n] = model[:n, :n]
    return out

def normalize(pow, model, mask):
    '''
    '''
    
    import scipy.ndimage
    
    
    mask, n = scipy.ndimage.measurements.label(mask)
    for i in xrange(1, n+1):
        sel = mask==i
        pow[sel] = ndimage_utility.histeq(pow[sel])
        model[sel] = ndimage_utility.histeq(model[sel])


def change_contrast(value, contrast):
    ''' Change pixel contrast by some factor
    '''
    
    return min(max(((value-127) * contrast / 100) + 127, 0), 255)

def write_ctf(raw, freq, roo, bg_window, ctf_output, apix, **extra):
    ''' Write a CTF file for CTFMatch
    '''
    
    roo1 = ctf.factor_correction(roo, 0, len(roo))[0]
    bg = ctf.background(raw, bg_window)
    km1 = 1.0 / (2.0*apix)
    dk = km1/len(roo)
    freq = freq*dk
    
    data = numpy.vstack((freq, bg, roo, roo1))
    format.write(ctf_output, data.T, header='freq,roo,bgsub,bgenv'.split(','))

#bg_window

def color_powerspectra(pow, roo, roo1, freq, label, defocus, rng, mask_radius, peaks=None, raw=None, peak_snr=1, peak_rng=[1,10], **extra):
    '''Create enhanced 2D power spectra
    '''
    
    if plotting.is_plotting_disabled(): return pow
    if mask_radius > 0:
        mask_radius=int(ctf.resolution(mask_radius, len(roo), **extra))
        pow *= ndimage_utility.model_disk(mask_radius, pow.shape)*-1+1
    pow = ndimage_utility.replace_outlier(pow, 3, 3, replace='mean')
    fig, ax=plotting.draw_image(pow, label=label, **extra)
    newax = ax.twinx()
    
    
    if extra['cs'] == 0.0:
        model = ctf.ctf_model_orig(defocus, freq, len(roo), **extra)**2
    else:
        model = ctf.ctf_model_spi(defocus, freq, len(roo), **extra)**2
    
    if True:
        linestyle = '-'
        linecolor = 'r'
    else:
        linestyle = ':'
        linecolor = 'w'
        
    freq = freq[rng[0]:]
    model = model[rng[0]:]
    model -= model.min()
    model /= model.max()
    roo1 -= roo1.min()
    roo1 /= roo1.max()
    
    if 1 == 0:
        import scipy.signal
        tmp = model[:rng[1]-rng[0]]
        #tmp = roo1
        #roo1=raw
        tmp = raw
        peaks = scipy.signal.find_peaks_cwt(tmp, numpy.arange(*peak_rng), min_snr=peak_snr)
        #peaks = scipy.signal.argrelmin(tmp, order=3)
        
        print len(peaks), raw.shape[0], roo1.shape[0]
    tmp=roo1
        
    newax.plot(freq[:rng[1]-rng[0]]+len(roo), model[:rng[1]-rng[0]], c=linecolor, linestyle=linestyle)
    newax.plot(freq[:len(roo1)]+len(roo), roo1, c='w')
    err = numpy.abs(roo1-model[:rng[1]-rng[0]])
    err = ctf.subtract_background(err, 27)
    newax.plot(freq[:rng[1]-rng[0]]+len(roo), err+1.1, c='y')
    if peaks is not None:
        newax.plot(freq[peaks]+len(roo), tmp[peaks], c='g', linestyle='None', marker='+')
        
    if 1 == 0:
        import scipy.signal
        diffraw = numpy.diff(numpy.log(raw-raw.min()+1))
        cut = numpy.max(scipy.signal.find_peaks_cwt(-1*diffraw, numpy.arange(1,10), min_snr=2))
        newax.plot(freq[cut]+len(roo), tmp[cut], c='b', linestyle='None', marker='o')
        cut = numpy.max(scipy.signal.find_peaks_cwt(-1*diffraw[:cut], numpy.arange(1,10), min_snr=2))
        newax.plot(freq[cut]+len(roo), tmp[cut], c='b', linestyle='None', marker='*')
    
    res = numpy.asarray([ctf.resolution(f, len(roo), **extra) for f in freq[:len(roo1)]])
    
    
    apix = extra['apix']
    for v in numpy.logspace(numpy.log10(apix*2), numpy.log10(20), 4):
        i = numpy.argmin(numpy.abs(v-res))
        newax.text(freq[i]+len(roo), -0.5, "%.1f"%res[i], color='black', backgroundcolor='white', fontsize=4)
        newax.plot([freq[i]+len(roo), freq[i]+len(roo)], [-0.5, 0], ls='-', c='y')
    
    val = numpy.max(numpy.abs(roo1))*4
    newax.set_ylim(-val, val)
    newax.set_axis_off()
    newax.get_yaxis().tick_left()
    newax.axes.get_yaxis().set_visible(False)
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    newax.get_yaxis().tick_left()
    newax.axes.get_yaxis().set_visible(False)
    #ax.set_xlim(-pow.shape[0]/2, pow.shape[0]/2)
    #ax.set_xlim(0, pow.shape[0])
    #print ax.get_xlim()
    #plotting.trim()
    return fig
    #return plotting.save_as_image(fig)

def generate_powerspectra(filename, bin_factor, invert, window_size, overlap, pad, offset, from_power=False, cache_pow=False, pow_cache="", disable_average=False, multitaper=False, rmax=9, biased=False, trans_file="", frame_beg=0, frame_end=-1, decimate_to=256, **extra):
    ''' Generate a power spectra using a perdiogram
    
    :Parameters:
    
    filename : str
               Input filename
    bin_factor : float
                Decimation factor
    invert : bool
             Invert the contrast
    window_size : int
                  Perdiogram window size
    overlap : float
              Amount of overlap between windows
    pad : int
          Number of times to pad the perdiogram
    from_power : bool
                 Is the input file already a power spectra
    
    :Returns:
    
    pow : array
          2D power spectra
    '''
    
    pow_cache = spider_utility.spider_filename(pow_cache, filename)
    if cache_pow:
        if os.path.exists(pow_cache):
            return ndimage_file.read_image(pow_cache)
    
    step = max(1, window_size*overlap)
    if from_power:
        return ndimage_file.read_image(filename)
    elif ndimage_file.count_images(filename) > 1 and disable_average:
        pow = None
        _logger.info("Average power spectra over individual frames")
        n = ndimage_file.count_images(filename)
        for i in xrange(n):
            mic = ndimage_file.read_image(filename, i).astype(numpy.float32)
            if bin_factor > 1: mic = ndimage_interpolate.downsample(mic, bin_factor)
            if invert: ndimage_utility.invert(mic, mic)
            rwin = ndimage_utility.rolling_window(mic[offset:mic.shape[0]-offset, offset:mic.shape[1]-offset], (window_size, window_size), (step, step))
            rwin = rwin.reshape((rwin.shape[0]*rwin.shape[1], rwin.shape[2], rwin.shape[3]))
            if 1 == 1:
                if pow is None: pow = ndimage_utility.powerspec_avg(rwin, pad)/float(n)
                else: pow += ndimage_utility.powerspec_avg(rwin, pad)/float(n)
            else:
                if pow is None:
                    pow, total = ndimage_utility.powerspec_sum(rwin, pad)
                else:
                    _logger.error("%d -- %f"%(total, pow.sum()))
                    total+= ndimage_utility.powerspec_sum(rwin, pad, pow, total)[1]
        pow=ndimage_utility.powerspec_fin(pow, total)
    else:
        if ndimage_file.count_images(filename) > 1 and not disable_average:
            trans = format.read(trans_file, numeric=True, spiderid=filename) if trans_file != "" else None
            mic = None
            if trans is not None:
                if frame_beg > 0: frame_beg -= 1
                if frame_end == -1: frame_end=len(trans)
                for i in xrange(frame_beg, frame_end):
                    frame = ndimage_file.read_image(filename, i)
                    j = i-frame_beg if len(trans) == (frame_end-frame_beg) else i
                    assert(j>=0)
                    frame = ndimage_utility.fourier_shift(frame, trans[j].dx, trans[j].dy)
                    if mic is None: mic = frame
                    else: mic += frame
            else:
                for i in xrange(ndimage_file.count_images(filename)):
                    frame = ndimage_file.read_image(filename, i)
                    if mic is None: mic = frame
                    else: mic += frame
        else: 
            mic = ndimage_file.read_image(filename)
        mic = mic.astype(numpy.float32)
        if bin_factor > 1: mic = ndimage_interpolate.downsample(mic, bin_factor)
        if invert: ndimage_utility.invert(mic, mic)
        #window_size /= bin_factor
        #overlap_norm = 1.0 / (1.0-overlap)
        if multitaper:
            _logger.info("Estimating multitaper")
            if window_size > 0:
                n=window_size/2
                c = min(mic.shape)/2
                mic=mic[c-n:c+n, c-n:c+n]
            pow = ndimage_utility.multitaper_power_spectra(mic, rmax, biased)
            #if window_size > 0:
            #    pow = ndimage_interpolate.downsample(pow, float(pow.shape[0])/window_size)
        else:
            
            pow, nwin = ndimage_utility.perdiogram(mic, window_size, pad, overlap, offset, ret_more=True)
            _logger.info("Averaging over %d windows"%nwin)
            
            '''
            assert(numpy.alltrue(numpy.isfinite(pow)))
            if rwin.shape[0] == 1 and decimate_to > 0:
                pow = ndimage_interpolate.downsample(pow, pow.shape[0]/decimate_to)
            assert(numpy.alltrue(numpy.isfinite(pow)))
            '''
    if cache_pow:
        ndimage_file.write_image(pow_cache, pow)
        #plotting.draw_image(ndimage_utility.dct_avg(mic, pad), cmap=plotting.cm.hsv, output_filename=format_utility.add_prefix(pow_cache, "dct_"))
        #pylab.imshow(ccp, cm.jet)
        #ndimage_file.write_image(, )
    return pow
#(nptwo,fx,nf,ntap,spec,kopt)
#nptwo,fx,nf,ktop,spec,kopt
def sine_psd(img, ntap, pad=2):
    '''
    '''
    import scipy.fftpack
    
    
    nf = img.shape[0]
    np2 = img.shape[0]*pad
    img = ndimage_utility.pad_image(img.astype(numpy.complex64), (int(img.shape[0]*pad), int(img.shape[1]*pad)), 'm')
    pow = scipy.fftpack.fft2(img)
    spec = numpy.zeros(pow.shape, dtype=numpy.float32)
    
    if 1 == 1:
        print pow.shape, spec.shape, pow.dtype, spec.dtype
        ndimage_utility._image_utility.sine_psd(pow, spec, ntap, pad)
    else:
        ck = 1.0/float(ntap)**2
        klim=ntap
        for m in xrange(nf):
            m2 = 2*m
            for mb in xrange(nf):
                mb2 = 2*mb
                for k in xrange(klim):
                    for k2 in xrange(klim):
                        j1 = numpy.mod(m2+np2-(k+1), np2)
                        j2 = numpy.mod(m2+(k+1), np2)
                        j1b = numpy.mod(mb2+np2-(k2+1), np2)
                        j2b = numpy.mod(mb2+(k2+1), np2)
                        zz = pow[j1,j2]-pow[j1b,j2b]
                        wk = 1.0 - ck*float(k)**2
                        wk2 = 1.0 - ck*float(k2)**2
                        spec[m,mb] += ( ( zz.real**2 + zz.imag**2 ) * wk *wk2 )
                spec[m, mb]  *= (6.0*float(klim))/float(4*klim**2+3*klim-1)
    return spec
    
    
def initialize(files, param):
    # Initialize global parameters for the script
    
    if mpi_utility.is_root(**param):
        _logger.info("Input: %s"%( "Micrograph" if not param['from_power'] else "Power spec"  ))
        #_logger.info("Output: %s"%( "Color" if param['color'] else "Standard"  ))
        _logger.info("Bin-factor: %f"%param['bin_factor'])
        _logger.info("Invert: %f"%param['invert'])
        _logger.info("Window size: %f"%param['window_size'])
        _logger.info("Pad: %f"%param['pad'])
        _logger.info("Overlap: %f"%param['overlap'])
        _logger.info("Plotting disabled: %d"%plotting.is_plotting_disabled())
        _logger.info("Microscope parameters")
        _logger.info(" - Voltage: %f"%param['voltage'])
        _logger.info(" - CS: %f"%param['cs'])
        _logger.info(" - AmpCont: %f"%param['ampcont'])
        _logger.info(" - Pixel size: %f"%param['apix'])
        
        if param['cs'] == 0.0:
            _logger.info("Using CTF model appropriate for 0 CS")
        if param['select'] != "":
            select = format.read(param['select'], numeric=True)
            files = selection_utility.select_file_subset(files, select)
        param['defocus_arr'] = numpy.zeros((len(files), 7))
        try:
            param['defocus_val'] = format_utility.map_object_list(format.read(param['defocus_file'], numeric=True, header=param['defocus_header']))
        except:param['defocus_val']={}
        
        pow_cache = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'pow_orig', 'pow_'), param['output'])
        if not os.path.exists(os.path.dirname(pow_cache)): os.makedirs(os.path.dirname(pow_cache))
        param['pow_cache'] = pow_cache
        
        pow_color = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'pow_color', 'pow_'), param['output'])
        if not os.path.exists(os.path.dirname(pow_color)): os.makedirs(os.path.dirname(pow_color))
        param['pow_color'] = pow_color
        
        ctf_output = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'ctf', 'ctf_'), param['output'])
        if not os.path.exists(os.path.dirname(ctf_output)): os.makedirs(os.path.dirname(ctf_output))
        param['ctf_output'] = ctf_output
        
        summary_output = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'pow_sum', 'pow'), param['output'])
        if not os.path.exists(os.path.dirname(summary_output)): os.makedirs(os.path.dirname(summary_output))
        param['summary_output'] = summary_output
        pow_cache = spider_utility.spider_filename(pow_cache, files[0])
        if param['cache_pow'] and os.path.exists(pow_cache):
            _logger.info("Using cached power spectra: %s -- %s"%(pow_cache, pow_color))
    return sorted(files)

def reduce_all(filename, file_completed, defocus_arr, defocus_val, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    filename, vals = filename
    
    if len(vals) > 0:
        defocus_arr[file_completed-1, :5]=vals
        try:
            defocus_arr[file_completed-1, 5] = defocus_val[int(vals[0])].defocus
        except:
            defocus_arr[file_completed-1, 5]=0
            defocus_arr[file_completed-1, 6]=0
        else:
            try:
                defocus_arr[file_completed-1, 6] = defocus_val[int(vals[0])].astig_mag
            except:
                defocus_arr[file_completed-1, 6] = 0
    
    return filename

def finalize(files, defocus_arr, output, summary_output, good_file="", **extra):
    # Finalize global parameters for the script
    
    if len(files) > 3:
        plotting.plot_scatter(summary_output, defocus_arr[:, 1], "Defocus", defocus_arr[:, 5], "Bench", defocus_arr[:, 2])
        plotting.plot_scatter(summary_output, defocus_arr[:, 1], "Defocus", defocus_arr[:, 2], "Error")
        plotting.plot_scatter(summary_output, defocus_arr[:, 1], "Defocus", defocus_arr[:, 3], "Rank")
        plotting.plot_scatter(summary_output, defocus_arr[:, 1], "Defocus", defocus_arr[:, 4], "Resolution")
        plotting.plot_scatter(summary_output, defocus_arr[:, 2], "Error", defocus_arr[:, 3], "Rank")
        plotting.plot_scatter(summary_output, defocus_arr[:, 4], "Resolution", defocus_arr[:, 3], "Rank")
        plotting.plot_histogram_cum(summary_output, defocus_arr[:, 3], 'Rank', 'Micrographs')

    idx = numpy.argsort(defocus_arr[:, 2])[::-1]
    format.write(output, defocus_arr[idx], prefix="sel_", format=format.spiderdoc, header="id,defocus,error,rank,resolution,defocus_spi,astig".split(','))
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import OptionGroup 
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input filenames containing micrographs, window stacks or power spectra", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for defocus file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        
        pgroup.add_option("",   disable_color=False, help="Disable output of color 2d power spectra")
        pgroup.add_option("-s", select="",           help="Selection file")
        pgroup.add_option("",   dpi=100,              help="Resolution in dots per inche for color figure images")
        
        group = OptionGroup(parser, "Benchmarking", "Options to control benchmarking",  id=__name__)
        group.add_option("-d", defocus_file="", help="File containing defocus values")
        group.add_option("",   defocus_header="id:0,defocus:1", help="Header for file containing defocus values")
        group.add_option("-g", good_file="",    help="Gold standard benchmark")
        pgroup.add_option_group(group)
        
        parser.change_default(log_level=3)
    
    group = OptionGroup(parser, "Power Spectra Creation", "Options to control power spectra creation",  id=__name__)
    group.add_option("", invert=False, help="Invert the contrast - used for unprocessed CCD micrographs")
    group.add_option("", window_size=256, help="Size of the window for the power spec (pixels)")
    #group.add_option("", decimate_to=256, help="Experimental parameter used for single fft")
    group.add_option("", pad=2.0, help="Number of times to pad the power spec")
    group.add_option("", overlap=0.5, help="Amount of overlap between windows")
    group.add_option("", offset=0, help="Offset from the edge of the micrograph (pixels)")
    group.add_option("", from_power=False, help="Input is a powerspectra not a micrograph")
    group.add_option("", bg_window=21, help="Size of the background subtraction window (pixels)")
    group.add_option("", cache_pow=False, help="Save 2D power spectra")
    group.add_option("", trunc_1D=False, help="Place trunacted 1D/model on color 2D power spectra")
    group.add_option("", peak_snr=1.0, help="SNR for peak selection")
    group.add_option("", peak_rng=[1,10], help="Range for peak selection")
    #group.add_option("", background=False, help="Do not")
    group.add_option("", mask_radius=0,  help="Mask the center of the color power spectra (Resolution in Angstroms)")
    group.add_option("", disable_average=False,  help="Average power spectra not frames")
    group.add_option("", frame_beg=0,              help="Range for the number of frames")
    group.add_option("", frame_end=-1,             help="Range for the number of frames")
    group.add_option("", trans_file="",  help="Translations for individual frames")
    group.add_option("", multitaper=False,  help="Multi-taper PSD estimation")
    group.add_option("", experimental=False,  help="Experimental mode")
    group.add_option("", rmax=9,              help="Maximum resolution for multitaper")
    group.add_option("", biased=False,              help="Biased multitaper")
    
    pgroup.add_option_group(group)
    
def setup_main_options(parser, group):
    # Setup options for the main program
    
    parser.change_default(bin_factor=2)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if options.param_file == "": raise OptionValueError('SPIDER Params file empty')
    options.peak_rng = [int(v) for v in options.peak_rng]

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Determine defocus and select good power spectra
                        
                        http://
                        
                        $ ara-autoctf mic_*.ter -p params.ter -o defocus.ter
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=True,
        use_version = True,
    )

def dependents(): return [spider_params]
if __name__ == "__main__": main()

