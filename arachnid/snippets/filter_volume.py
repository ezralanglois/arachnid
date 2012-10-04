''' Filter a volume

Download to edit and run: :download:`plot_fsc.py <../../arachnid/snippets/filter_volume.py>`

To run:

.. sourcecode:: sh
    
    $ python filter_volume.py

.. literalinclude:: ../../arachnid/snippets/filter_volume.py
   :language: python
   :lines: 16-
   :linenos:
'''

#from ..core.metadata import spider_params, spider_utility
from ..core.spider import spider

if __name__ == '__main__':
    
    input_volume = "vol.spi"
    output_volume = "filt_vol"
    pixel_size = 1.2
    resolution = 20
    incore_mode = 1
    
    spi = spider.open_session([input_volume], spider_path="", thread_count=0, enable_results=False) # Create a SPIDER session using the extension of the input_volume
    
    if incore_mode == 1: 
        spi.fq(input_volume, spi.GAUS_LP, filter_radius=pixel_size/resolution, outputfile=output_volume) # Filter input_volume and write to output_volume
    elif incore_mode == 2:
        involume = spi.cp(input_volume)                                                     # Read volume to an incore file
        filtvolume = spi.fq(involume, spi.GAUS_LP, filter_radius=pixel_size/resolution)     # Filter input_volume and return incore incore filtered volume
        spi.cp(filtvolume, outputfile=output_volume)                                        # Write incore filtered volume to a file
    
    
    
    
    