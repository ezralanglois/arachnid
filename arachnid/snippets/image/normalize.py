''' Normalize images

Download to edit and run: :download:`normalize.py <../../arachnid/snippets/normalize.py>`

To run:

.. sourcecode:: sh
    
    $ python normalize.py

.. literalinclude:: ../../arachnid/snippets/normalize.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
sys.path.append('/Users/robertlanglois/workspace/arachnida/src/')
#sys.path.append('/home/robertl/tmp/arachnid-0.0.1/')
from arachnid.core.metadata import spider_utility
from arachnid.core.image import ndimage_file, ndimage_utility
import logging, glob, shutil, numpy


if __name__ == '__main__':

    # Parameters
    
    stack_files = glob.glob("/Users/robertlanglois/Desktop/win_0000686.dat")
    output_file = "tmp_00000.spi"
    inplace=False
    radius = 56
    test=True
    
    logging.basicConfig(level_level=logging.INFO)
    
    mask = None
    if test: #112/2
        for stack_file in stack_files:
            for i, img in enumerate(ndimage_file.iter_images(stack_file)):
                if mask is None:
                    mask = ndimage_utility.model_disk(radius, img.shape)*-1+1
                m = numpy.mean(img[mask>0.5])
                s = numpy.std(img[mask>0.5])
                if not numpy.allclose(m, 0.0) or not numpy.allclose(s, 1.0):
                    logging.error("m=%f, s=%f, i=%d"%(m, s, i))
                    assert(False)
    else:
        for stack_file in stack_files:
            logging.info("Processing: %s"%stack_file)
            if not inplace:
                output_file = spider_utility.spider_filename(output_file, stack_file)
            for i, img in enumerate(ndimage_file.iter_images(stack_file)):
                if mask is None:
                    mask = ndimage_utility.model_disk(radius, img.shape)*-1+1
                img = ndimage_utility.normalize_standard(img, mask)
                ndimage_file.write_image(output_file, img, i)
            if inplace: shutil.move(output_file, stack_file)
            
    