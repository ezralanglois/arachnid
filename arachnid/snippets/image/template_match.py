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
from arachnid.core.metadata import format_utility, spider_utility
from arachnid.core.image import ndimage_file, ndimage_utility, eman2_utility, manifold

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

def remove_overlap(scoords, radius, sel=None):
    '''
    '''
    
    if sel is None: sel = numpy.ones(len(scoords), dtype=numpy.bool)
    coords = scoords[:, 1:3].copy()
    i=0
    radius *= 1.1
    idx = numpy.argwhere(sel).squeeze()
    
    k=30
    neigh = manifold.knn(coords, k)
    dist = neigh.data.reshape((len(scoords), k+1))
    col = neigh.col.reshape((len(scoords), k+1))
    radius*=radius
    while i < len(idx):
        j=idx[i]
        jcol = col[j, 1:]
        jdist = dist[j, 1:]
        osel = jdist[sel[jcol]] < radius
        jcol = jcol[sel[jcol]][osel]
        if numpy.sum(osel) > 0:
            if numpy.alltrue(scoords[j, 0] > scoords[jcol, 0]):
                sel[jcol]=0
                idx = numpy.argwhere(sel).squeeze()
            else:
                sel[j]=0
                i+=1
        else:
            i+=1
    
    '''
    while i < len(idx):
        dist = scipy.spatial.distance.cdist(coords[idx[i+1:]], coords[idx[i]].reshape((1, len(coords[idx[i]]))), metric='euclidean').ravel()
        osel = dist < radius
        if numpy.sum(osel) > 0:
            if numpy.alltrue(scoords[idx[i], 0] > scoords[idx[i+1:], 0]):
                sel[idx[i+1:][osel]]=0
                idx = numpy.argwhere(sel).squeeze()
            else:
                sel[idx[i]]=0
        else:
            i+=1
    '''
    return sel

if __name__ == '__main__':

    # Parameters
    
    micrograph_file = sys.argv[1]   # micrograph_file = "mics/mic_*.spi"
    output_file=sys.argv[2] if len(sys.argv) > 4 else "box_000000.png"       # select_file="selected_mic_00000.spi"
    
    logging.basicConfig(level=3)
    bin_factor = 4.0
    window_size = int(312/bin_factor)
    radius = ((0.5*220)/bin_factor)
    invert=False
    bin_factor = 1.0
    
    template = ndimage_utility.model_disk(radius, (int(window_size/bin_factor), int(window_size/bin_factor)))
    files = glob.glob(micrograph_file)
    print "Running on %d files"%len(files)
    for i,filename in enumerate(files):
        print filename, i+1, len(files)
        sys.stdout.flush()

        output_file=spider_utility.spider_filename(output_file, filename)
        mic = ndimage_file.spider_writer.read_image(filename)
        mic = eman2_utility.gaussian_high_pass(mic, 0.25/radius, True)
        
        
        if bin_factor > 1.0:
            mic=eman2_utility.decimate(mic, bin_factor)
        if invert: ndimage_utility.invert(mic, mic)
        #mic = eman2_utility.gaussian_high_pass(mic, 0.25/radius, True)
        
        oradius = radius*1.2
        
        cradius = radius
        cc_map = ndimage_utility.cross_correlate(mic, template)
        boxmap=mic
        
        mic = eman2_utility.gaussian_low_pass(mic, 2.0/radius, True)
        
        for fradius in [0.0, 2, radius, radius/2, 4.8]:
            coords = ndimage_utility.find_peaks_fast(cc_map, oradius, fradius)
        
            pngmic=box_part_on_micrograph(boxmap, coords, window_size, bin_factor)
            pngmic.save(format_utility.add_prefix(output_file, '%f_'%fradius))
            print fradius, ':', len(coords), remove_overlap(coords, cradius).sum()
        
      
