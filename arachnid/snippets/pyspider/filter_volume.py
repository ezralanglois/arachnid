''' Volume filter

This example script shows how to filter a volume using the PySPIDER interface. This
script can be used to lowpass filter a volume or, alternatively, bandpass filter
the volume using SPIDER commands.

Note that the code contains a "comment", a line starting with a `#`. This 
comment explains the code on that line. 

While this script is included with the installed package, it is intended
to be edited before being run. It is recommended you download it from
the link below.

:download:`Download script <../../arachnid/snippets/pyspider/filter_volume.py>`

.. seealso::

    - List of :py:class:`SPIDER Commands <arachnid.core.spider.spider.Session>`
    - :py:func:`open_session <arachnid.core.spider.spider.open_session>`
    - :py:meth:`FQ <arachnid.core.spider.spider.Session.fq>`
    - :py:meth:`CP <arachnid.core.spider.spider.Session.cp>`

Requirements
    
    - :doc:`Installation <../install>` of the Arachnid Python package
    - Installation of the SPIDER package (binary)
        
        - Download: http://www.wadsworth.org/spider_doc/spider/docs/spi-register.html
        - Install: http://www.wadsworth.org/spider_doc/spider/docs/installation.html

To run:

.. sourcecode:: sh
    
    $ python filter_volume.py

.. literalinclude:: ../../arachnid/snippets/pyspider/filter_volume.py
   :language: python
   :lines: 42-
   :linenos:
'''
from arachnid.core.spider import spider

if __name__ == '__main__':                      # This line is not necessary for script execution, but helps when building the documentation
    
    input_volume = "input_volume.spi"           # Filename for input file (should have extension)
    output_volume = "filtered_output_volume"    # Filename for output file (extension optional)
    pixel_size = 1.2                            # Pixel size of the volume
    resolution = 20                             # Resolution to lowpass filter the volume
    

    # *** Uncomment the following to filter from 80 to 20 angstroms ***
    #resolution = (80, 20)  
    
    # Create a SPIDER session using the extension of the input_volume
    #    - Alternatively, you can specify the extension with data_ext="dat" for the .dat extension
    #    - If no input file is given and no extension specified, the default is "spi"
    #    - Note that, if you specify an extension then this overrides the extension of the input file
    spi = spider.open_session([input_volume], spider_path="", thread_count=0, enable_results=False, data_ext="")
    
    # Test whether to perform a band pass filter
    if not isinstance(resolution, tuple): 
        # Filter the volume using the Gaussian lowpass filter to the specified `resolution` with the given `pixel_size`
        spi.fq(input_volume, spi.GAUS_LP, filter_radius=pixel_size/resolution, outputfile=output_volume) # Filter input_volume and write to output_volume
    else:
        # Filter the volume using the Butterworth highpass and Gaussian lowpass filter to the specified high and low `resolution` with the given `pixel_size`
        involume = spi.cp(input_volume)                                                      # Read volume to an incore file
        filtvolume = spi.fq(involume, spi.BUTER_LP, filter_radius=pixel_size/resolution[0])  # Highpass filter incore volume with Butterworth
        filtvolume = spi.fq(filtvolume, spi.GAUS_HP, filter_radius=pixel_size/resolution[1]) # Lowpass filter incore volume with Gaussian
        spi.cp(filtvolume, outputfile=output_volume)                                         # Write incore filtered volume to a file
        
        # The above example can be shortened to two lines as follows
        if 1 == 0:                                                                                             # Do not run the following code - for illustrative purposes
            filtvolume = spi.fq(involume, spi.BUTER_LP, filter_radius=pixel_size/resolution[0])                # Highpass filter incore volume with Butterworth
            spi.fq(filtvolume, spi.GAUS_HP, filter_radius=pixel_size/resolution[1], outputfile=output_volume)  # Lowpass filter incore volume with Gaussian
            
    
    
    
    