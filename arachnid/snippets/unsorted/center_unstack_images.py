''' Center a set of images using parameters from an alignment file

Download to edit and run: :download:`center_images.py <../../arachnid/snippets/center_images.py>`

To run:

.. sourcecode:: sh
    
    $ python center_images.py

.. literalinclude:: ../../arachnid/snippets/center_images.py
   :language: python
   :lines: 16-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file, ndimage_utility
from arachnid.core.orient import orient_utility
import numpy, itertools

if __name__ == '__main__':

    # Parameters
    
    align_file = ""
    image_file = ""
    output_file = ""
    
    # Read an alignment file
    align = format.read_alignment(align_file)
    align,header = format_utility.tuple2numpy(align)
    
    align[:, 16]-=1
    iter_single_images = ndimage_file.iter_images(image_file, align[:, 15:17])
    #align[:, 0] = -align[:, 5]
        
    psi = -numpy.deg2rad(align[:, 5])
    ca = numpy.cos(psi)
    sa = numpy.sin(psi)
    tx = align[:, 6]*ca + align[:, 7]*sa
    ty = align[:, 7]*ca - align[:, 6]*sa
    
    print tx[0], ty[0]
    print orient_utility.align_param_2D_to_3D(numpy.deg2rad(align[0, 5]), align[0, 6], align[0, 7])
        
    iter_single_images = itertools.imap(ndimage_utility.fourier_shift, iter_single_images, tx, ty)
    for i, img in enumerate(iter_single_images):
        ndimage_file.write_image(spider_utility.spider_filename(output_file, int(align[i, 4])), img)

