''' This script grabs a set number of projections per view

Download to edit and run: :download:`uniform_view_subset.py <../../arachnid/snippets/uniform_view_subset.py>`

To run:

.. sourcecode:: sh
    
    $ python uniform_view_subset.py

.. literalinclude:: ../../arachnid/snippets/uniform_view_subset.py
   :language: python
   :lines: 16-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility
from arachnid.core.image import ndimage_file
import numpy

if __name__ == '__main__':

    # Parameters
    
    align_file = ""
    stack_file = ""
    output_file = ""
    particle_count=5000
    
    
    # Read an alignment file
    align, header = format_utility.tuple2numpy(format.read_alignment(align_file))
    view = align[:, 3].astype(numpy.int)
    views = numpy.unique(view)
    ids = []
    for v in views:
        ids.extend(numpy.argwhere(v==view)[:particle_count])
    ids = numpy.asarray(ids)
    
    for i, img in enumerate(ndimage_file.iter_images(stack_file, ids)):
        ndimage_file.write_image(output_file, img, i)
    
    print ids.shape, align[ids].shape, header
    format.write(output_file, align[ids], prefix="align_", header=header, default_format=format.spiderdoc)