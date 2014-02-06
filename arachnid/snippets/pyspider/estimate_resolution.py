''' Estimate the resolution between two half volumes

This example script shows how to use SPIDER to estimate the resolution between
two half volumes.

Note that the code contains a "comment", a line starting with a `#`. This 
comment explains the code on that line. 

While this script is included with the installed package, it is intended
to be edited before being run. It is recommended you download it from
the link below.

:download:`Download script <../../arachnid/snippets/estimate_resolution.py>`

.. seealso::

    - List of :py:class:`SPIDER Commands <arachnid.core.spider.spider.Session>`
    - :py:func:`open_session <arachnid.core.spider.spider.open_session>`
    - :py:meth:`RF 3 <arachnid.core.spider.spider.Session.rf_3>`

Requirements
    
    - :doc:`Installation <install>` of the Arachnid Python package
    - Installation of the SPIDER package (binary)
        
        - Download: http://www.wadsworth.org/spider_doc/spider/docs/spi-register.html
        - Install: http://www.wadsworth.org/spider_doc/spider/docs/installation.html

To run:

.. sourcecode:: sh
    
    $ python estimate_resolution.py

.. literalinclude:: ../../arachnid/snippets/estimate_resolution.py
   :language: python
   :lines: 40-
   :linenos:
'''
from arachnid.core.metadata import spider_params
from arachnid.core.spider import spider

if __name__ == '__main__':
    
    input_eve_volume = "raw_even.spi"
    input_odd_volume = "raw_odd"
    output_resolution = "dres"
    
    params_file="params"    # Set this to the spider params file
    pixel_size = 1.0        # If there is a params file, do not set this
    
    # Create a SPIDER session using the extension of the input_volume
    #    - Alternatively, you can specify the extension with data_ext="dat" for the .dat extension
    #    - If no input file is given and no extension specified, the default is "spi"
    #    - Note that, if you specify an extension then this overrides the extension of the input file
    spi = spider.open_session([input_eve_volume], spider_path="", thread_count=0, enable_results=False, data_ext="")
    if params_file != "":
        # Read data from a SPIDER params file
        params = spider_params.read(params_file)
        pixel_size = params['apix']
    
    # Estimate the resolution between two half volumes
    dum,pres,sp = spi.rf_3(input_eve_volume, input_odd_volume, outputfile=output_resolution)
    print "Resolution:", pixel_size/sp
    
    
    

