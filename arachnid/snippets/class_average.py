''' Generate a view average from an aligned stack and alignment file

Download to edit and run: :download:`class_average.py <../../arachnid/snippets/class_average.py>`

To run:

.. sourcecode:: sh
    
    $ python class_average.py

.. literalinclude:: ../../arachnid/snippets/class_average.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1/')

from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file, eman2_utility
import numpy #, itertools
eman2_utility;

if __name__ == '__main__':

    # Parameters
    
    align_file = "align_dala_025.int"
    image_file = "dala_025.int"
    output_file = "avg_001.int"
    
    align = format.read_alignment(align_file)
    align, header = format_utility.tuple2numpy(align)
    refidx = header.index('ref_num')
    label = numpy.zeros((len(align), 2), dtype=numpy.int)
    label[:, 0] = spider_utility.spider_id(image_file)
    label[:, 1] = align[:, 4].astype(numpy.int)-1
    
    ref = align[:, refidx].astype(numpy.int)
    refs = numpy.unique(ref)
    
    for i, view in enumerate(refs):
        avg = None
        for img in ndimage_file.iter_images(image_file, label[ref==view]):
            if avg is None: avg = img
            else: avg += img
        tot = numpy.sum(ref==view)
        print i, view, tot
        ndimage_file.write_image(output_file, avg/tot, i)
   


