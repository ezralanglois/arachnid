''' Plot boxes on a micrograph image

Download to edit and run: :download:`box_micrograph.py <../../arachnid/snippets/box_micrograph.py>`

To run: --- 1, 411 -- /data2/amedee/43S-eIF4

.. sourcecode:: sh
    
    $ python box_micrograph.py

.. literalinclude:: ../../arachnid/snippets/box_micrograph.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.metadata import format, spider_utility
from arachnid.core.image import ndimage_file, eman2_utility, ndimage_utility

try: 
    from PIL import ImageDraw 
    ImageDraw;
except: import ImageDraw
import scipy.misc, glob
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
    
    micrograph_file = sys.argv[1]
    coord_file=sys.argv[2]
    output_file=sys.argv[3]
    
    bin_factor = 1.0
    window_size = 234
    radius = ((0.5*112)/bin_factor)
    invert=True
    
    files = glob.glob(micrograph_file)
    for i,filename in enumerate(files):
        print filename, i+1, len(files)
        sys.stdout.flush()

        output_file=spider_utility.spider_filename(output_file, filename)
        coord_file=spider_utility.spider_filename(coord_file, filename)
        mic = ndimage_file.read_image(filename)
        if bin_factor > 1.0:
            mic=eman2_utility.decimate(mic, bin_factor)
        if invert: ndimage_utility.invert(mic, mic)
        #mic = eman2_utility.gaussian_high_pass(mic, 0.25/radius, True)
        
        coords = format.read(coord_file, numeric=True)
        
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