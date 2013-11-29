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
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.metadata import format, spider_utility
from arachnid.core.image import ndimage_file, ndimage_interpolate, ndimage_utility #, ndimage_filter

try: 
    from PIL import ImageDraw #@UnresolvedImport
    ImageDraw;
except: import ImageDraw
import scipy.misc, glob, os, logging
#, pylab

def box_part_on_micrograph(mic, coords, window_size, bin_factor):
    '''
    '''
    
    mic = scipy.misc.toimage(mic).convert("RGB")
    draw = ImageDraw.Draw(mic)
    
    width=int((window_size / float(bin_factor))*0.5)
    for box in coords:
        x = box.x / bin_factor
        y = box.y / bin_factor
        draw.rectangle((x+width, y+width, x-width, y-width), fill=None, outline="#ff4040")
    return mic

if __name__ == '__main__':

    # Parameters
    
    micrograph_file = sys.argv[1]   # micrograph_file = "mics/mic_*.spi"
    coord_file=sys.argv[2]          # coord_file="local/coords/sndc_00000.spi"
    output_file=sys.argv[3]         # output_file="boxed_mic_00000.spi"
    select_file=sys.argv[4] if len(sys.argv) > 4 else ""       # select_file="selected_mic_00000.spi"
    
    logging.basicConfig(level=3)
    bin_factor = 8.0
    window_size = 256
    radius = ((0.5*200)/bin_factor)
    invert=True
    
    files = glob.glob(micrograph_file)
    print "Running on %d files"%len(files)
    for i,filename in enumerate(files):
        print filename, i+1, len(files)
        sys.stdout.flush()

        output_file=spider_utility.spider_filename(output_file, filename)
        coord_file=spider_utility.spider_filename(coord_file, filename)
        if not os.path.exists(coord_file):
            logging.warn("Skipping - no coordinate file: %s"%coord_file)
            continue
        mic = ndimage_file.spider.read_image(filename)
        ndimage_utility.replace_outlier(mic, 2.5, out=mic)
        if bin_factor > 1.0:
            mic=ndimage_interpolate.downsample(mic, bin_factor)
        if invert: ndimage_utility.invert(mic, mic)
        #mic = ndimage_filter.gaussian_highpass(mic, 0.25/radius, True)
        
        coords = format.read(coord_file, numeric=True)
        if select_file != "":
            select_file=spider_utility.spider_filename(select_file, filename)
            if not os.path.exists(select_file): 
                logging.warn("Skipping - no selection file: %s"%select_file)
                continue
            select = format.read(select_file, numeric=True)
            print 'Selecting', len(select), 'of', len(coords)
            coords = [coords[s.id-1] for s in select]
        
        mic=box_part_on_micrograph(mic, coords, window_size, bin_factor)
        mic.save(output_file)
    
    '''
    dpi=200
    fig = pylab.figure(0, dpi=dpi, facecolor=facecolor)
    ax = pylab.axes(frameon=False)
    ax.set_axis_off()
    ax.imshow(img, cmap=pylab.cm.gray)
    
    pylab.fill([3,4,4,3], [2,2,4,4], 'b', alpha=0.2, edgecolor='r')
    '''