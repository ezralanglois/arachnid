''' Test if frame cropping worked

Download to edit and run: :download:`test_frame_crop.py <../../arachnid/snippets/test_frame_crop.py>`

To run:

.. sourcecode:: sh
    
    $ python test_frame_crop.py

.. literalinclude:: ../../arachnid/snippets/test_frame_crop.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.app import tracing
from arachnid.core.image import ndimage_file, ndimage_interpolate
from arachnid.core.metadata import spider_utility
import numpy,logging, glob

if __name__ == "__main__":
    
    image_file = sys.argv[1]
    frame_file = sys.argv[2]
    index = int(sys.argv[3])
    findex = int(sys.argv[4])
    
    tracing.configure_logging(log_level=3)
    frames = glob.glob(spider_utility.spider_filename(frame_file, image_file))
    
    ref = ndimage_file.read_image(image_file, index-1)
    avg = None
    for f in frames:
        img = ndimage_file.read_image(f, findex-1)
        if img.shape[0] > ref.shape[0]:
            img = ndimage_interpolate.interpolate(img, float(img.shape[0])/ref.shape[0])
        if avg is None: avg = img
        else: avg += img
    
    #avg /= len(frames)
    
    logging.info("Difference: %f"%numpy.sqrt(numpy.sum(numpy.square(avg-ref))))
    
    
    