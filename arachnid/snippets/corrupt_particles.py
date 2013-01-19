''' Find and remove corrupt particles using a selection file

When you see the following message from relion, then run this script and use the new selection file to run relion again

.. sourcecode:: sh

    Check for dimension: 1
    MultidimArray shape:  Check for dimension: 1
    MultidimArray shape:  Empty MultidimArray!
    
    Check called from file ./src/multidim_array.h line 3329
     Empty MultidimArray!


Download to edit and run: :download:`corrupt_particles.py <../../arachnid/snippets/corrupt_particles.py>`

To run:

.. sourcecode:: sh
    
    $ python corrupt_particles.py

.. literalinclude:: ../../arachnid/snippets/corrupt_particles.py
   :language: python
   :lines: 28-
   :linenos:
'''
from arachnid.core.metadata import format, spider_utility
from arachnid.core.image.formats import mrc
from arachnid.core.image import ndimage_file
import numpy, logging


if __name__ == '__main__':
    input_selection = ""
    output_selection=""
    mrc_stack=False
    
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    
    logging.info("Reading input selection file")
    isel = format.read(input_selection)
    sel = []
    
    logging.info("Copying image stack")
    batch = len(isel)/100
    for i, s in enumerate(isel):
        
        logging.info("Reading image: %d"%(i+1))
        #if (i%batch) == 0:
        #    logging.info("Copying image stack - %f%% done"%((i/len(isel)*100)))
        filename, offset = spider_utility.relion_file(s.rlnImageName)
        if mrc_stack:
            np = mrc.read_image(filename, offset-1)
        else:
            np = ndimage_file.read_image(filename, offset-1)
        if numpy.isfinite(numpy.mean(np)):
            sel.append(s)
    
    logging.info("Writing output selection file")
    format.write(output_selection, sel, default_format=format.spiderdoc)
    logging.info("Completed")