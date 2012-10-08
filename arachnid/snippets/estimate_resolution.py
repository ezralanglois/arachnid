''' Estimate the resolution between two half volumes

Download to edit and run: :download:`estimate_resolution.py <../../arachnid/snippets/estimate_resolution.py>`

.. seealso::

    List of :py:class:`SPIDER Commands <arachnid.core.spider.spider.Session>`

To run:

.. sourcecode:: sh
    
    $ python estimate_resolution.py

.. literalinclude:: ../../arachnid/snippets/estimate_resolution.py
   :language: python
   :lines: 21-
   :linenos:
'''

from arachnid.core.image import ndimage_file
from arachnid.core.metadata import spider_params
from arachnid.core.spider import spider

if __name__ == '__main__':
    
    input_eve_volume = "raw_even.spi"
    input_odd_volume = "raw_odd.spi"
    output_eve_volume = "masked_raw_even.spi"
    output_odd_volume = "masked_raw_odd.spi"
    output_resolution = "dres"
    volume_mask = ""
    
    params_file="params"
    pixel_size = 1.2        # If there is a params file, do not set this
    pixel_diameter = 200    # If there is a params file, do not set this
    
    # Invoke a SPIDER session using the extension from input_volume
    spi = spider.open_session([input_eve_volume], spider_path="", thread_count=0, enable_results=False) # Create a SPIDER session using the extension of the input_volume
    if params_file != "":
        params = spider_params.read(params_file)
        pixel_size = params['apix']
        pixel_diameter = params['pixel_diameter']
    
    if volume_mask != "":
        img = ndimage_file.read_image(spi.replace_ext(input_eve_volume))
        img = ndimage_file.read_image(spi.replace_ext(input_odd_volume))
        mask = ndimage_file.read_image(spi.replace_ext(volume_mask))
        ndimage_file.write_image(spi.replace_ext(output_eve_volume), img*mask) 
        ndimage_file.write_image(spi.replace_ext(output_odd_volume), img*mask) 
        input_eve_volume=output_eve_volume
        input_odd_volume=output_odd_volume
    
    dum,pres,sp = spi.rf_3(input_eve_volume, input_odd_volume, outputfile=output_resolution)
    
    print "Resolution:", pixel_size/sp
    
    
    

