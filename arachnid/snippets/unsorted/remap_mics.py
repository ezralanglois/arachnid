''' Remap micrographs to original filename

Download to edit and run: :download:`remap_mics.py <../../arachnid/snippets/remap_mics.py>`

To run:

.. sourcecode:: sh
    
    $ python remap_mics.py

.. literalinclude:: ../../arachnid/snippets/remap_mics.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.app import tracing
from arachnid.core.metadata import format, spider_utility, format_utility
from arachnid.core.image import ndimage_file, ndimage_utility, ndimage_filter
import logging, glob, numpy

if __name__ == "__main__":

    tracing.configure_logging(log_level=3)
    if 1 == 1:
        coord_file1 = sys.argv[1]
        coord_file2 = sys.argv[2]
        map_file = sys.argv[3]
        output = sys.argv[4]
        known = {}
        
        legmap = format.read(map_file, numeric=True)
        legmap = format_utility.map_object_list(legmap, 'araSpiderID')
        for filename in glob.glob(coord_file2):
            ref=format.read(filename, ndarray=True)[0][:, 2:]
            if len(ref) not in known: known[len(ref)]=[]
            known[len(ref)].append((filename, ref))
        
        total_zero=0
        vals=[]
        for filename in glob.glob(coord_file1):
            coords=format.read(filename, ndarray=True)[0][:, 2:]
            best = (1e20, None)
            for ref_file, ref in known[len(coords)]:
                dist = numpy.sum(numpy.square(coords-ref))
                if dist < best[0]: best=(dist, ref_file)
            id = spider_utility.spider_id(filename)
            leg = legmap[spider_utility.spider_id(best[1])].araLeginonFilename
            vals.append((leg, id))
            if best[0]==0: total_zero+=0
            logging.error("%s -> %d for %s"%(str(best), id, leg))
        
        logging.error("Found %d of %d"%(total_zero, len(glob.glob(coord_file1))))
        format.write(output, vals, header="araLeginonFilename,araSpiderID".split(','))    
        
    elif 1 == 1:
        image_file1 = sys.argv[1]
        image_file2 = sys.argv[2]
        coord_file = sys.argv[3]
        output = sys.argv[4]
        stacks = glob.glob(image_file2)
        vals = []
        
        for i, stack in enumerate(stacks):
            logging.info("Cropping: %s"%stack)
            mic = ndimage_file.read_image(stack)
            win = ndimage_utility.crop_window(mic, 1000, 1000, 234/2)
            ndimage_file.write_image('tmp_stack.spi', win, i)
        
        for filename in glob.glob(image_file1):
            mic = ndimage_file.read_image(filename)
            coords=format.read(spider_utility.spider_filename(coord_file, filename), numeric=True)
            ref = ndimage_utility.crop_window(mic, 1000, 1000, 234/2)
            best = (1e20, None)
            for win in ndimage_file.iter_images('tmp_stack.spi'):
                dist = numpy.sum(numpy.square(ref-win))
                if dist < best[0]: best = (dist, stack)
            id = spider_utility.spider_id(filename)
            vals.append((best[1], id))
            logging.info("%s -> %d for %f"%(filename, id, best[0]))
        format.write(output, vals, header="araLeginonFilename,araSpiderID".split(','))
    elif 1 == 1:
        image_file = sys.argv[1]
        stack_file = sys.argv[2]
        output = sys.argv[3]
        stacks = glob.glob(stack_file)
        vals = []
        norm_mask=ndimage_utility.model_disk(90, (234,234))
        norm_mask=norm_mask*-1+1
        for filename in glob.glob(image_file):
            ref = ndimage_file.read_image(filename)
            ndimage_utility.normalize_standard(ref, norm_mask, out=ref)
            best = (1e20, None)
            for stack in stacks:
                win = ndimage_file.read_image(stack)
                ndimage_utility.normalize_standard(win, norm_mask, out=win)
                dist = numpy.sum(numpy.square(ref-win))
                if dist < best[0]: best = (dist, stack)
            id = spider_utility.spider_id(best[1])
            vals.append((filename, id))
            logging.info("%s -> %d for %f"%(filename, id, best[0]))
        format.write(output, vals, header="araLeginonFilename,araSpiderID".split(','))
    else:
        image_file = sys.argv[1]
        coord_file = sys.argv[2]
        stack_file = sys.argv[3]
        output = sys.argv[4]
    
        tracing.configure_logging(log_level=3)
        norm_mask=ndimage_utility.model_disk(90, (234,234))
        norm_mask=norm_mask*-1+1
        stacks = glob.glob(stack_file)
        vals = []
        for filename in glob.glob(image_file):
            mic = ndimage_file.read_image(filename)
            ndimage_utility.invert(mic, mic)
            mic = ndimage_filter.gaussian_highpass(mic, 1.0/(234.0), True)
            best = (1e20, None)
            for stack in stacks:
                win = ndimage_file.read_image(stack)
                coords=format.read(spider_utility.spider_filename(coord_file, stack), numeric=True)
                img = ndimage_utility.crop_window(mic, coords[0].x, coords[0].y, win.shape[0]/2)
                win=ndimage_filter.ramp(win)
                ndimage_utility.replace_outlier(win, 2.5, out=win)
                ndimage_utility.normalize_standard(win, norm_mask, out=win)
                dist = numpy.sum(numpy.square(img-win))
                if dist < best[0]: best = (dist, stack)
            id = spider_utility.spider_id(best[1])
            vals.append((filename, id))
            logging.info("%s -> %d for %f"%(filename, id, best[0]))
        format.write(output, vals, header="araLeginonFilename,araSpiderID".split(','))
        