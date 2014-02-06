''' Test the SPIDER reprojection binding

Download to edit and run: :download:`reproject_test.py <../../arachnid/snippets/reproject_test.py>`

To run:

.. sourcecode:: sh
    
    $ python reproject_test.py vol.spi ref_stack.spi 2

.. literalinclude:: ../../arachnid/snippets/reproject_test.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys
from arachnid.core.image import ndimage_file
from arachnid.core.image import reproject
from arachnid.core.orient import healpix

if __name__ == '__main__':

    # Parameters
    image_file = sys.argv[1]
    output = sys.argv[2]
    healpix_order = int(sys.argv[3])
    
    vol = ndimage_file.read_image(image_file)
    rad = vol.shape[0]/2
    ang = healpix.angles(healpix_order)
    imgs = reproject.reproject_3q(vol, rad, ang)
    ndimage_file.write_stack(output, imgs)
