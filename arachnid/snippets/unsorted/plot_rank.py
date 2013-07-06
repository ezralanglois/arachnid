''' Plot the rank column in a SPIDER alignment file

Download to edit and run: :download:`plot_rank.py <../../arachnid/snippets/plot_rank.py>`

To run:

.. sourcecode:: sh
    
    $ python plot_rank.py vol.spi ref_stack.spi 2

.. literalinclude:: ../../arachnid/snippets/plot_rank.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
import matplotlib
matplotlib.use("Agg")
from arachnid.core.metadata import format, format_utility
import pylab

if __name__ == "__main__":
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    align, header = format.read_alignment(input_file, ndarray=True)
    m = header.index('mirror')
    
    pylab.hist(align[:, m])
    pylab.savefig(format_utility.new_filename(output_file, ext="png"), dpi=200)
    
    