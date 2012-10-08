''' Mask a volume

Download to edit and run: :download:`mask_volume.py <../../arachnid/snippets/mask_volume.py>`

.. seealso::

    List of :py:class:`SPIDER Commands <arachnid.core.spider.spider.Session>`

To run:

.. sourcecode:: sh
    
    $ python mask_volume.py

.. literalinclude:: ../../arachnid/snippets/mask_volume.py
   :language: python
   :lines: 20-
   :linenos:
'''
from arachnid.core.image import ndimage_utility, ndimage_file
from arachnid.core.spider import spider

if __name__ == '__main__':
    
    input_volume = "vol.spi"
    output_volume = "filt_vol.spi"
    mask_type = "tight"
    mask_edge_width = 3
    mask_edge_type = 'C'
    pixel_diameter = 200
    
    if mask_type == "tight":
        img = ndimage_file.read_image(input_volume)                                                     # Read volume from a file
        mask, th = ndimage_utility.tight_mask(img, threshold=None, ndilate=1, gk_size=3, gk_sigma=9)    # Generate a adaptive tight mask
        print "Threshold: ", th                                                                         # Print threshold used to create the tight mask
        ndimage_file.write_image(output_volume, img*mask)                                               # Write the masked volume to output_volume
    elif mask_type == "spherical":
        spi = spider.open_session([input_volume], spider_path="", thread_count=0, enable_results=False) # Create a SPIDER session using the extension of the input_volume
        center = spider.image_size(spi, input_volume)[0]/2+1                                                                   # Center of the volume
        radius = pixel_diameter/2+mask_edge_width/2 if mask_edge_type == 'C' else pixel_diameter/2+mask_edge_width                # Define radius of the mask
        spi.ma(input_volume, radius, (center, center, center), mask_edge_type, 'C', mask_edge_width, outputfile=output_volume) # Create a spherical mask with a soft edge
    
    