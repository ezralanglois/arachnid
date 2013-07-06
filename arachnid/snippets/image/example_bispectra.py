''' Generate image of example bispectra of a particle

Download to edit and run: :download:`example_bispectra.py <../../arachnid/snippets/example_bispectra.py>`

    
Right click on link -> "Copy Link Address" or "Copy Link Location"
Download from the command line:

.. sourcecode:: sh
    
    $ wget <paste-url-here>

How to run command:

.. sourcecode:: sh
    
    $ python example_bispectra.py "particle_*.spi" bispec_0000.png

.. literalinclude:: ../../arachnid/snippets/example_bispectra.py
   :language: python
   :lines: 23-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.metadata import spider_utility
from arachnid.core.image import ndimage_file, ndimage_utility
import glob #, os, logging

if __name__ == '__main__':

    # Parameters
    
    image_file = sys.argv[1]   # micrograph_file = "mics/mic_*.spi"
    output_file=sys.argv[2]         # output_file="boxed_mic_00000.spi"
    
    for filename in glob.glob(image_file):
        img = ndimage_file.read_image(filename)
        bimg, freq = ndimage_utility.bispectrum(img, img.shape[0]-1, 'gaussian')
        ndimage_file.write_image(spider_utility.spider_filename(output_file, filename), bimg.real)