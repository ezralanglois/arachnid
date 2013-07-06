''' Plots an FSC that can be customized for publication

See http://matplotlib.org/ for more details on customizing plotting.

Download to edit and run: :download:`plot_fsc.py <../../arachnid/snippets/plot_fsc.py>`

To run:

.. sourcecode:: sh
    
    $ python plot_fsc.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/plot_fsc.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
import matplotlib
matplotlib.use("Agg")
from arachnid.core.metadata import format, spider_utility
import pylab, os, numpy

if __name__ == '__main__':

    # Parameters
    select_file1 = sys.argv[1]
    select_file2 = sys.argv[2]
    output = sys.argv[3]
    dpi=1000
    
    # Script
    
    
    # Read a resolution file
    select1 = format.read(select_file1, numeric=True)
    select2 = format.read(select_file2, numeric=True)
    print select_file1
    print select_file2
    
    id1 = numpy.zeros(len(select1), dtype=numpy.int)
    for i, v in enumerate(select1):
        id1[i] = spider_utility.relion_id(v.rlnImageName)[0]
    id2 = numpy.zeros(len(select2), dtype=numpy.int)
    for i, v in enumerate(select2):
        id2[i] = spider_utility.relion_id(v.rlnImageName)[0]
    mics = numpy.unique( numpy.concatenate((id1, id2)) )
    count = numpy.zeros((len(mics), 2), dtype=numpy.int)
    for i, m in enumerate(mics):
        count[i, 0] = numpy.sum(m==id1)
        count[i, 1] = numpy.sum(m==id2)
    
    frac = numpy.zeros(len(count))
    #sel = count[:, 0]>count[:, 1]
    frac = count[:, 0].astype(numpy.float)/(count[:, 0]+count[:, 1])
    idx = numpy.argsort(count[:, 0]-count[:, 1])
    count = count[idx]
    mics = mics[idx]
    
    if 1 == 1:
        pylab.figure(figsize=(15, 6))
        pylab.hist(frac, 256)
        
    elif 1 == 1:
        pylab.figure(figsize=(20, 6))
        n = numpy.arange(len(mics))
        width=0.2
        n=n[:20]
        count=count[:20]
        pylab.bar(n, count[:, 0], width, color='crimson', label=select_file1)
        pylab.bar(n+width, count[:, 1], width, color='burlywood', label=select_file2)
        #pylab.hist(count, len(count), histtype='bar', color=['crimson', 'burlywood'], label=[select_file1, select_file2])
        pylab.legend()
    
    if 1 == 0:
        n = int(numpy.sqrt(len(count)))
        H, xedges, yedges = numpy.histogram2d(count[:, 0], count[:, 1], bins=(n, n))
        pylab.imshow(H, extent=[yedges[0], yedges[-1], xedges[-1], xedges[0]], interpolation='nearest')
        pylab.colorbar()
    pylab.savefig(os.path.splitext(output)[0]+".png", dpi=dpi)


