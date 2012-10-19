''' Find and remove corrupt particles using a selection file

Download to edit and run: :download:`corrupt_particles.py <../../arachnid/snippets/corrupt_particles.py>`

To run:

.. sourcecode:: sh
    
    $ python corrupt_particles.py

.. literalinclude:: ../../arachnid/snippets/corrupt_particles.py
   :language: python
   :lines: 16-
   :linenos:
'''
from arachnid.core.metadata import format, spider_utility
from arachnid.core.image import ndimage_file
import numpy


if __name__ == '__main__':
    input_selection = ""
    stack_file = ""
    output_selection=""
    
    isel = format.read(input_selection)
    sel = []
    
    for s in isel:
        
        filename, offset = spider_utility.relion_file(s.rlnImageName)
        np = ndimage_file.read_image(filename, offset-1)
        if numpy.isfinite(numpy.mean(np)):
            sel.append(s)
    
    format.write(output_selection, sel)