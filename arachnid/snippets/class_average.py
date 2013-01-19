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
from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file, eman2_utility
import numpy #, itertools
eman2_utility;

if __name__ == '__main__':

    # Parameters
    
    align_file = "data/align_03.sdo"
    image_file = "data/dala03.sdo"
    output_file = "avg_001.spi"
    
    align = format.read_alignment(align_file)
    align, header = format_utility.tuple2numpy(align)
    refidx = header.index('ref_num')
    label = numpy.zeros((len(align), 2), dtype=numpy.int)
    label[:, 0] = spider_utility.spider_id(image_file)
    label[:, 1] = align[:, 4].astype(numpy.int)-1
    
    ref = align[:, refidx].astype(numpy.int)
    refs = numpy.unique(ref)
    sel = refs[0] == ref
    
    label = label[sel]
    align = align[sel]
    
    avg = None
    for i in xrange(label.shape[0]):
        img = ndimage_file.read_image(spider_utility.spider_filename(image_file, int(label[i, 0])), int(label[i, 1]))
        print align[i, 1], align[i, refidx]
        if align[i, 1] >= 180.0: img[:,:] = eman2_utility.mirror(img)
        if avg is None: avg = numpy.zeros(img.shape)
        avg += img
    ndimage_file.write_image(output_file, avg)


