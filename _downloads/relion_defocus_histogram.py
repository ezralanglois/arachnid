''' Create a histogram of particles per defocus for a relion selection file

Download to edit and run: :download:`relion_defocus_histogram.py <../../arachnid/snippets/metadata/relion_defocus_histogram.py>`

To run:

.. sourcecode:: sh
    
    $ python relion_defocus_histogram.py

.. literalinclude:: ../../arachnid/snippets/metadata/relion_defocus_histogram.py
   :language: python
   :lines: 16-
   :linenos:
'''
from arachnid.core.util.matplotlib_nogui import pylab
import sys
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1/')
from arachnid.core.metadata import format, format_utility
import numpy

if __name__ == '__main__':

    # Parameters
    
    input_file = sys.argv[1] # relion.star
    output_file = sys.argv[2]
    dpi=72
    
    vals = format.read(input_file, numeric=True)
    vals = numpy.asarray([v.rlnDefocusU for v in vals])
    fig = pylab.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=numpy.sqrt(vals.shape[0]))
    pylab.xlabel('Defocus')
    pylab.ylabel('Number of Particles')
    
    fig.savefig(format_utility.new_filename(output_file, ext="png"), dpi=dpi)
