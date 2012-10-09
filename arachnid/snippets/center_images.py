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
from arachnid.core.image import ndimage_file, eman2_utility
import numpy

if __name__ == '__main__':

    # Parameters
    
    align_file = ""
    image_file = ""
    output_file = ""
    
    # Read an alignment file
    align = format.read_alignment(align_file)
    align,header = format_utility.tuple2numpy(align)
    
    for i, img in enumerate(ndimage_file.iter_images(image_file, align[:, 15:17])):
        
        # Convert TR to RT
        psi = -align[i].psi
        ca = numpy.cos(align[i].psi)
        sa = numpy.sin(align[i].psi)
        x = align[i].tx*ca + align[i].ty*sa
        y = align[i].ty*ca - align[i].tx*sa
        
        #Shift the image
        img = eman2_utility.fshift(img, x, y)
        
        #Write to single image file
        ndimage_file.write_image(spider_utility.spider_filename(output_file, int(align[i, 4])), img)

