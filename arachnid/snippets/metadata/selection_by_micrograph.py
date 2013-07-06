''' This script repacks a selection for for a full stack into selection files by micrograph

Download to edit and run: :download:`selection_by_micrograph.py <../../arachnid/snippets/selection_by_micrograph.py>`

To run:

.. sourcecode:: sh
    
    $ python selection_by_micrograph.py

.. literalinclude:: ../../arachnid/snippets/selection_by_micrograph.py
   :language: python
   :lines: 16-
   :linenos:
'''
#import sys
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1/')
from arachnid.core.metadata import format
import numpy

if __name__ == '__main__':

    # Parameters
    
    full_select = "clean3_bispec/sel_clean_006.dat"
    mic_map_select = "data/new_order_select.ter"
    poutput_file = ""
    noutput_file = "refinement/data/bispec_select/select_00000.ter"
    align_file="data/align_05.ter"


    align = format.read_alignment(align_file, numeric=True)
    id = numpy.asarray(align)[:, 4].max()
    selected = numpy.zeros(id, dtype=numpy.bool)
    select = numpy.asarray(format.read(full_select, numeric=True))
    smap = format.read(mic_map_select, numeric=True, header="id,count,defocus,a,b,micrograph".split(","))
    select = select[:, 0].astype(numpy.int)-1
    selected[select]=1
    beg = 0
    for sm in smap:
        end = sm.count+beg
        if poutput_file != "":
            total = numpy.sum(selected[beg:end])
            select = numpy.hstack((numpy.argwhere(selected[beg:end]).reshape((total, 1))+1, numpy.ones(total).reshape((total, 1))))
            format.write(poutput_file, select, spiderid=sm.micrograph, format=format.spidersel,header="id,select".split(','))
        if noutput_file != "":
            total = numpy.sum(numpy.logical_not(selected[beg:end]))
            if total > 0:
                select = numpy.hstack((numpy.argwhere(numpy.logical_not(selected[beg:end])).reshape((total, 1))+1, numpy.ones(total).reshape((total, 1))))
                format.write(noutput_file, select, spiderid=sm.micrograph, format=format.spidersel,header="id,select".split(','))
        beg=end


        