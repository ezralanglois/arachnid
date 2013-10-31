''' 

Download to edit and run: :download:`multitaper_experiment.py <../../arachnid/snippets/multitaper_experiment.py>`

To run:

.. sourcecode:: sh
    
    $ python multitaper_experiment.py

.. literalinclude:: ../../arachnid/snippets/multitaper_experiment.py
   :language: python
   :lines: 17-
   :linenos:
'''

import sys,numpy

from arachnid.core.metadata import spider_params #, spider_utility
from arachnid.core.image import ndimage_file, ndimage_utility, ctf
from arachnid.core.util import plotting
import scipy.signal
scipy.signal;


if __name__ == '__main__':
    mic = sys.argv[1]
    params = sys.argv[2]
    output = sys.argv[3]
    pad=1.0
    overlap=0.5
    offset=0
    biased=True
    
    
    mic = ndimage_file.read_image(mic)
    extra = spider_params.read(params)
    
    mic = mic #[:1024,:1024]
    rmin=30.0
    rmax2=5.0
    peak_idx=None
    
    min_freq = extra['apix']/rmin
    max_freq = extra['apix']/rmax2
    
    '''
    
    if rmin < rmax: rmin, rmax = rmax, rmin
    min_freq = extra['apix']/rmin
    max_freq = extra['apix']/rmax
    models = []
    test = numpy.zeros(4)
    for scale in (1, 0.76, 0.88, 0.64):
        coefs = demodulate(pow, eps_phase, window_phase, scale*min_freq*pow.shape[0], scale*max_freq*pow.shape[0])
    '''
    
    '''
    import scipy.signal
        diffraw = numpy.diff(numpy.log(raw-raw.min()+1))
        cut = numpy.max(scipy.signal.find_peaks_cwt(-1*diffraw, numpy.arange(1,10), min_snr=2))
        newax.plot(freq[cut]+len(roo), tmp[cut], c='b', linestyle='None', marker='o')
    '''
    
    if 1 == 1:
        # Perdiogram
        for window_size in [min(mic.shape), 512, 256]:
            pow, nwin = ndimage_utility.perdiogram(mic, window_size, pad, overlap, offset, ret_more=True)
            
            
            rang = ((max_freq*pow.shape[0])-(min_freq*pow.shape[0]))/2.0
            freq2 = float(pow.shape[0])/(rang/1)
            freq1 = float(pow.shape[0])/(rang/15)
            pow = ndimage_utility.filter_annular_bp(pow, freq1, freq2)
            #pow = ndimage_utility.spiral_transform(pow)
            
            #n = pow.shape[0]/2
            #val = 0.5/(ctf.resolution(6, n, **extra)-ctf.resolution(30, n, **extra))
            #pow = ndimage_filter.filter_gaussian_highpass(pow, val)
            print 'Perdiogram', nwin
            raw = ndimage_utility.mean_azimuthal(pow)[:pow.shape[0]/2]
            raw[1:] = raw[:len(raw)-1]
            raw[:2]=0
            print 'raw', len(raw)
            st = int(ctf.resolution(rmin, len(raw), **extra))
            e = int(ctf.resolution(rmax2, len(raw), **extra)) if rmax2 > 0 else len(raw)
            peak_idx = scipy.signal.find_peaks_cwt(raw[st:e], numpy.arange(1,10), min_snr=2)
            plotting.plot_line('perdiogram_%d_rng.png'%window_size, [ctf.resolution(i, len(raw), **extra) for i in xrange(st, e)], raw[st:e], dpi=300, marker_idx=peak_idx)
    
    if 1==1:
        # Multitaper
        for rmax in [3, 5]:#, 7, 9, 12, 15, 17]:
            pow = ndimage_utility.multitaper_power_spectra(mic, rmax, biased)
            
            rang = ((max_freq*pow.shape[0])-(min_freq*pow.shape[0]))/2.0
            freq2 = float(pow.shape[0])/(rang/1)
            freq1 = float(pow.shape[0])/(rang/15)
            pow = ndimage_utility.filter_annular_bp(pow, freq1, freq2)
            #pow = ndimage_utility.spiral_transform(pow)
            
            print 'Multitaper', rmax 
            raw = ndimage_utility.mean_azimuthal(pow)[:pow.shape[0]/2]
            raw[1:] = raw[:len(raw)-1]
            raw[:2]=0
            st = int(ctf.resolution(rmin, len(raw), **extra))
            e = int(ctf.resolution(rmax2, len(raw), **extra)) if rmax2 > 0 else len(raw)
            #peak_idx = scipy.signal.find_peaks_cwt(raw[st:e], numpy.arange(1,10), min_snr=1)
            
            plotting.plot_line('multitaper_%d_rng.png'%rmax, [ctf.resolution(i, len(raw), **extra) for i in xrange(st, e)], raw[st:e]/raw[st:e].max(), dpi=300, marker_idx=peak_idx)
    
    
    