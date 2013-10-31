''' Plot boxes on a micrograph image

Download to edit and run: :download:`box_micrograph.py <../../arachnid/snippets/box_micrograph.py>`

    
Right click on link -> "Copy Link Address" or "Copy Link Location"
Download from the command line:

.. sourcecode:: sh
    
    $ wget <paste-url-here>

How to run command:

.. sourcecode:: sh
    
    $ python box_micrograph.py "mics/mic_*.spi" local/coords/sndc_00000.spi boxed_mic_00000.spi

.. literalinclude:: ../../arachnid/snippets/box_micrograph.py
   :language: python
   :lines: 23-
   :linenos:
'''
import sys
from arachnid.core.metadata import format_utility, spider_utility, spider_params
from arachnid.core.image import ndimage_file, ndimage_utility, manifold
from arachnid.core.image import ndimage_filter, ndimage_interpolate
try: 
    from PIL import ImageDraw 
    ImageDraw;
except: import ImageDraw
import scipy.misc, glob, logging, scipy.spatial.distance, numpy
#, pylab

def box_part_on_micrograph(mic, coords, window_size, bin_factor):
    '''
    '''
    
    mic = scipy.misc.toimage(mic).convert("RGB")
    draw = ImageDraw.Draw(mic)
    
    width=int((window_size / float(bin_factor))*0.5)
    for box in coords:
        if hasattr(box, 'x'):
            x = box.x / bin_factor
            y = box.y / bin_factor
        else:
            x = box[1] / bin_factor
            y = box[2] / bin_factor
        draw.rectangle((x+width, y+width, x-width, y-width), fill=None, outline="#ff4040")
    return mic

if __name__ == '__main__':

    # Parameters
    
    micrograph_file = sys.argv[1]   # micrograph_file = "mics/mic_*.spi"
    param_file = sys.argv[2]
    output_file=sys.argv[3] if len(sys.argv) > 3 else "box_000000.png"       # select_file="selected_mic_00000.spi"
    
    logging.basicConfig(level=3)
    bin_factor = 4.0
    param = spider_params.read(param_file)
    
    window_size = int(param['window']/bin_factor)
    radius = ((0.5*param['pixel_diameter'])/bin_factor)
    invert=True
    bin_factor = 1.0
    
    template = ndimage_utility.model_disk(radius, (int(window_size/bin_factor), int(window_size/bin_factor)))
    files = glob.glob(micrograph_file)
    print "Running on %d files"%len(files)
    for i,filename in enumerate(files):
        print filename, i+1, len(files)
        sys.stdout.flush()

        output_file=spider_utility.spider_filename(output_file, filename)
        mic = ndimage_file._default_write_format.read_image(filename)
        mic = ndimage_filter.gaussian_highpass(mic, 0.25/radius, 2)
        
        
        if bin_factor > 1.0:
            mic=ndimage_interpolate.downsample(mic, bin_factor)
        if invert: ndimage_utility.invert(mic, mic)
        
        oradius = radius
        cradius = radius
        boxmap = ndimage_utility.replace_outlier(mic, 3.0)
        #boxmap = ndimage_filter.gaussian_lowpass(boxmap, 2.0/radius, 2)
        
        
        cc_map = ndimage_utility.cross_correlate(mic, template)
        numpy.fabs(cc_map, cc_map)
        cc_map -= float(numpy.max(cc_map))
        cc_map *= -1
        
        mic = ndimage_filter.gaussian_lowpass(mic, 2.0/radius, 2)
        
        for fradius in [0.0, 2, radius, radius/2, 4.8]:
            coords = ndimage_utility.find_peaks_fast(cc_map, oradius, fradius)
            pngmic=box_part_on_micrograph(boxmap, coords, window_size, bin_factor)
            pngmic.save(format_utility.add_prefix(output_file, '%f_'%fradius))
            
        
      
