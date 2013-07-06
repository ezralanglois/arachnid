''' Extract selected projections and restack

Download to edit and run: :download:`restack_2d_class.py <../../arachnid/snippets/restack_2d_class.py>`

To run:

.. sourcecode:: sh
    
    $ python restack_2d_class.py

.. literalinclude:: ../../arachnid/snippets/restack_2d_class.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
sys;
from arachnid.core.metadata import format #, format_utility, spider_utility
from arachnid.core.image import ndimage_file
import numpy

if __name__ == '__main__':

    # Parameters
    
    image_file = sys.argv[1]
    select_file = sys.argv[2]
    cluster_file = sys.argv[3]
    output_file = sys.argv[4]
    
    selected = []
    selcluster = set(v.id for v in format.read(select_file, numeric=True))
    cluster = format.read(cluster_file, numeric=True)
    
    for c in cluster:
        if c.ref_num in selcluster:
            selected.append(c.id-1)
    
    selected = numpy.asarray(selected, dtype=numpy.int)
    for i, img in enumerate(ndimage_file.iter_images(image_file, selected)):
        ndimage_file.write_image(output_file, img, i)
