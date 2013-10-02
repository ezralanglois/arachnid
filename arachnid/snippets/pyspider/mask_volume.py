''' Volume mask

This example script shows how to apply a soft spherical mask to a volume using the 
PySPIDER interface.

Note that the code contains a "comment", a line starting with a `#`. This 
comment explains the code on that line. 

While this script is included with the installed package, it is intended
to be edited before being run. It is recommended you download it from
the link below.

:download:`Download script <../../arachnid/snippets/pyspider/mask_volume.py>`

.. seealso::

    - List of :py:class:`SPIDER Commands <arachnid.core.spider.spider.Session>`
    - :py:func:`open_session <arachnid.core.spider.spider.open_session>`
    - :py:func:`image_size <arachnid.core.spider.spider.image_size>`
    - :py:meth:`image_size <arachnid.core.spider.spider.Session.ma>`

To run:

.. sourcecode:: sh
    
    $ python mask_volume.py

.. literalinclude:: ../../arachnid/snippets/pyspider/mask_volume.py
   :language: python
   :lines: 33-
   :linenos:
'''
from arachnid.core.spider import spider

if __name__ == '__main__':           # This line is not necessary for script execution, but helps when building the documentation
    
    input_volume = "vol.spi"         # Filename for input file (should have extension)
    output_volume = "filt_vol.spi"   # Filename for output file (extension optional)
    mask_edge_width = 3              # Mask width
    mask_edge_type = 'C'             # Soften spherical mask with cosine edge
    pixel_diameter = 200             # Diameter of particle (or mask)
    
    # Create a SPIDER session using the extension of the input_volume
    #    - Alternatively, you can specify the extension with data_ext="dat" for the .dat extension
    #    - If no input file is given and no extension specified, the default is "spi"
    #    - Note that, if you specify an extension then this overrides the extension of the input file
    spi = spider.open_session([input_volume], spider_path="", thread_count=0, enable_results=False, data_ext="")
    
    
    # Center of the volume
    #  - `image_size` is a convenience function that is not apart of the main SPIDER command set
    #  - `image_size` returns a tuple of 3 values (width, height, depth), the `[0]` gets the first value (depth)
    center = spider.image_size(spi, input_volume)[0]/2+1
    
    # Define radius of the mask using the diameter of the particle in pixels plus the width of the mask  
    radius = pixel_diameter/2+mask_edge_width
    
    # Creates and applies a spherical mask to the input volume with given `radius` and `center` soften the edge with a cosine function
    #  'C' ensures the background is an average along the mask border
    spi.ma(input_volume, radius, (center, center, center), mask_edge_type, 'C', mask_edge_width, outputfile=output_volume) 
    
    