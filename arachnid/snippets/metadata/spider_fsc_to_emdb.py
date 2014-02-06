''' Convert a SPIDER resolution file to an EMDB compatible XML file

Download to edit and run: :download:`spider_fsc_to_emdb.py <../../arachnid/snippets/spider_fsc_to_emdb.py>`

To run:

.. sourcecode:: sh
    
    $ python spider_fsc_to_emdb.py

.. literalinclude:: ../../arachnid/snippets/spider_fsc_to_emdb.py
   :language: python
   :lines: 16-
   :linenos:
'''
from arachnid.core.metadata import format
import numpy, os

def write_xml(output, x, y):
    '''
    '''
    fout = open(output, 'w')
    fout.write('<?xml version="1.0" encoding="UTF-8"?>\n') #header
    fout.write('<fsc title="%s" xaxis="%s" yaxis="%s">\n'%('FSC Plot', 'Normalized Spatial Frequency', 'Fourier Shell Correlation')) #FSC Tag
    
    for i in xrange(len(x)):
        fout.write('\t<coordinate>\n\t\t<x>%f</x>\n\t\t<y>%f</y>\n\t</coordinate>\n'%(x[i], y[i]))
    fout.write('</fsc>') #FSC Tag end
    fout.close()


if __name__ == '__main__':

    # Parameters
    
    input_file = ""
    output_file = ""
    
    vals = numpy.asarray(format.read(input_file, numeric=True, header="id,freq,dph,fsc,fscrit,voxels"))
    write_xml(os.path.splitext(output_file)[0]+'.xml', vals[:, 1], vals[:, 3])
