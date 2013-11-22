''' Create defocus groups and repack stack files

Download to edit and run: :download:`defocus_group.py <../../arachnid/snippets/defocus_group.py>`

To run:

.. sourcecode:: sh
    
    $ python defocus_group.py

.. literalinclude:: ../../arachnid/snippets/defocus_group.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1/')
from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file
import numpy

if __name__ == '__main__':

    # Parameters
    
    defocus_file = sys.argv[1]
    output_file=sys.argv[2]
    defocus_diff = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    stack_file = sys.argv[4] if len(sys.argv) > 4 else None
    write_stack = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    header = "id,defocus,astig_mag,astig_ang,cutoff".split(',')
    
    defocus, header = format.read(defocus_file, ndarray=True, header=header)
    print "Original Number of groups", defocus.shape[0]
    defocus_col = header.index('defocus')
    id_col = header.index('id')
    regroup = []
    idx = numpy.argsort(defocus[:, defocus_col])
    
    id_map = numpy.zeros((defocus.shape[0], 2)) if stack_file is None else numpy.zeros((defocus.shape[0], 3)) 
    start = 0
    while start < len(defocus):
        total=numpy.sum((defocus[idx[start:], defocus_col]-defocus[idx[start], defocus_col]) < defocus_diff)
        regroup.append(idx[start:start+total])
        id_map[start:start+total, 0] = len(regroup)
        id_map[start:start+total, 1] = defocus[idx[start:start+total], id_col]
        if stack_file is not None:
            for i in xrange(start, start+total):
                id_map[i, 2] = ndimage_file.count_images(spider_utility.spider_filename(stack_file, int(defocus[idx[i], id_col])))
        start += total
    
    if stack_file is not None:
        format.write(output_file, id_map, prefix="sel_", header="id,micrograph,total".split(','), default_format=format.spiderdoc)
    else:
        format.write(output_file, id_map, prefix="sel_", header="id,micrograph".split(','), default_format=format.spiderdoc)
    new_defocus = numpy.zeros((len(regroup), 2)) if stack_file is None else numpy.zeros((len(regroup), 3)) 
    start = 0
    for i, g in enumerate(regroup):
        new_defocus[i, 0] = i+1
        new_defocus[i, 1] = numpy.mean(defocus[g, defocus_col])
        if stack_file is not None:
            new_defocus[i, 2] = numpy.sum(id_map[start:start+len(g), 2])
            start += len(g)
    if stack_file is not None:
        format.write(output_file, new_defocus, prefix="defocus_", header="id,defocus,total".split(','), default_format=format.spiderdoc)
    else:
        format.write(output_file, new_defocus, prefix="defocus_", header="id,defocus".split(','), default_format=format.spiderdoc)
    
    print "New Number of groups", len(regroup)
    
    if stack_file is not None and write_stack == 1:
        for i, g in enumerate(regroup):
            output_file = spider_utility.spider_filename(output_file, i+1)
            index = 0
            for mic in g:
                for img in ndimage_file.iter_images(spider_utility.spider_filename(stack_file, int(defocus[mic, id_col]))):
                    ndimage_file.write_image(output_file, img, index)
                    index +=1
    