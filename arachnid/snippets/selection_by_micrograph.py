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
from arachnid.core.metadata import format
import numpy

if __name__ == '__main__':

    # Parameters
    
    full_select = ""
    mic_map_select = ""
    poutput_file = ""
    noutput_file = ""
    align_file=""

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
        total = numpy.sum(selected[beg:end])
        select = numpy.hstack((numpy.argwhere(selected[beg:end]).reshape((total, 1))+1, numpy.ones(total).reshape((total, 1))))
        format.write(poutput_file, select, spiderid=sm.micrograph, format=format.spidersel,header="id,select".split(','))
        total = numpy.sum(numpy.logical_not(selected[beg:end]))
        select = numpy.hstack((numpy.argwhere(numpy.logical_not(selected[beg:end])).reshape((total, 1))+1, numpy.ones(total).reshape((total, 1))))
        format.write(noutput_file, select, spiderid=sm.micrograph, format=format.spidersel,header="id,select".split(','))
        beg=end


        