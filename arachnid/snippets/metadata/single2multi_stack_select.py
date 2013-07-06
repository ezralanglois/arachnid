''' Convert single stack selection to multi-stack selection

Download to edit and run: :download:`single2multi_stack_select.py <../../arachnid/snippets/single2multi_stack_select.py>`

To run:

.. sourcecode:: sh
    
    $ python single2multi_stack_select.py

.. literalinclude:: ../../arachnid/snippets/single2multi_stack_select.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.metadata import format #, format_utility, spider_utility
import numpy

if __name__ == "__main__":
    
    select_file = sys.argv[1]
    cluster_file = sys.argv[2]
    stack_select = sys.argv[3]
    output_file = sys.argv[4]
    
    selected = []
    selcluster = set(v.id for v in format.read(select_file, numeric=True))
    cluster = format.read(cluster_file, numeric=True)
    stack_select = format.read(stack_select, ndarray=True)[0]
    
    for c in cluster:
        if c.ref_num in selcluster:
            selected.append(int(c.id-1))
    selected = numpy.asarray(selected, dtype=numpy.int)
    
    sel_by_mic={}
    for i in selected:
        sel_by_mic.setdefault(int(stack_select[i, 1]), []).append(int(stack_select[i, 2]))
    
    for id, sel in sel_by_mic.iteritems():
        sel = numpy.asarray(sel)
        format.write(output_file, numpy.vstack((sel, numpy.ones(sel.shape[0]))).T, spiderid=id, header=['id', 'select'], default_format=format.spidersel)
    
    
    
    