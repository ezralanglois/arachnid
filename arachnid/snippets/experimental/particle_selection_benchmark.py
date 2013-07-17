''' Benchmark the consistency of particle selection between two coordinate sets

Download to edit and run: :download:`particle_selection_benchmark.py <../../arachnid/snippets/particle_selection_benchmark.py>`

    
Right click on link -> "Copy Link Address" or "Copy Link Location"
Download from the command line:

.. sourcecode:: sh
    
    $ wget <paste-url-here>

How to run command:

.. sourcecode:: sh
    
    $ python particle_selection_benchmark.py 'coords/sndc*.spi' 'coords_good/sndc*.spi' plot.png

.. literalinclude:: ../../arachnid/snippets/particle_selection_benchmark.py
   :language: python
   :lines: 23-
   :linenos:
'''
import sys, os
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.util import bench
from arachnid.core.metadata import format, spider_utility,format_utility
bench;
import numpy #, logging
from arachnid.core.util.matplotlib_nogui import pylab

#format.mrccoord._logger.setLevel(logging.DEBUG)

def plot_bar(output, xdata, ydata, base, ylabel, xlabel):
    '''
    '''

    
    if not os.path.exists(output): os.makedirs(output)
    try:
        pylab.clf()
    except: print "Failed to plot histogram"
    
    fit = pylab.polyfit(xdata, ydata, 1)
    fit_fn = pylab.poly1d(fit)
    pylab.axis([xdata.min(), xdata.max(), 0.0,1.0])
    pylab.gca().set_autoscaley_on(False)
    pylab.plot(xdata, ydata, 'bo', xdata, fit_fn(xdata), '--k')
    print "Slope:", (fit_fn(xdata)[-1]-fit_fn(xdata)[0])/(xdata[-1]-xdata[0])
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.savefig(os.path.join(output, base), dpi=1000)

if __name__ == '__main__':

    # Parameters
    
    #10442 13661
    select = sys.argv[1]
    coords1_file = sys.argv[2]        # image_file = "data/dala_01.spi"
    coords2_file = sys.argv[3]        # align_file = "data/align_01.spi"
    defocus_file=sys.argv[4]           # output_file="stack01.spi"
    output=sys.argv[5]           # output_file="stack01.spi"
    pixel_radius = 100
    use_bins=True
    use_equal=True
    min_defocus=10000
    max_defocus=70000
    
    #logging.basicConfig(level=logging.DEBUG)
    
    n=0
    select = format.read(select, numeric=True)
    defocus_dict = format.read(defocus_file, numeric=True)
    for i in xrange(len(defocus_dict)-1, 0, -1):
        if defocus_dict[i].defocus < min_defocus or defocus_dict[i].defocus > max_defocus:
            print "Removing micrograph %d because defocus %f violates allowed range %f-%f "%(defocus_dict[i].id, defocus_dict[i].defocus, min_defocus, max_defocus)
            del defocus_dict[i]
    defvals = format_utility.map_object_list(defocus_dict)
    defmap = numpy.zeros((len(select), 4))
    for s in select:
        coords1 = format.read(coords1_file, spiderid=s.araSpiderID, numeric=True)
        if not os.path.exists(spider_utility.spider_filename(coords2_file, s.araLeginonFilename)):
            print '**', s.araLeginonFilename, s.araSpiderID, 0, len(coords1), 0, 0, spider_utility.spider_filename(coords2_file, s.araLeginonFilename)
            continue
        try:
            coords2 = format.read(coords2_file, spiderid=s.araLeginonFilename, numeric=True)
        except:
            print spider_utility.spider_filename(coords2_file, s.araLeginonFilename)
            raise
        
        benchmark = numpy.vstack(([c.x for c in coords2], [c.y for c in coords2])).T.astype(numpy.float)
        autop = numpy.vstack(([c.x for c in coords1], [c.y for c in coords1])).T.astype(numpy.float)
        overlap = bench.find_overlap(autop, benchmark, pixel_radius)
        id = spider_utility.spider_id(s.araSpiderID)
        if id not in defvals: continue
        defmap[n, :] = [defvals[id].defocus, len(coords1), len(benchmark), len(overlap)]
        print s.araLeginonFilename, s.araSpiderID, defmap[n, :]
        n+=1
        #            TP            FP                      TN            FN
        #tab = ( len(overlap), len(autop)-len(overlap),    0,  len(benchmark)-len(overlap) )
        #print s.araLeginonFilename, s.araSpiderID, " ".join([str(t) for t in tab])
    print "Number of mics:", n
    defmap = defmap[:n]
    if use_bins:
        tmp = defmap
        defmap = numpy.zeros((int(numpy.sqrt(tmp.shape[0])), tmp.shape[1]))
        if use_equal:
            counts = numpy.zeros(defmap.shape[0], dtype=numpy.int)
            for i in xrange(counts.shape[0]):
                counts[i] = ( (tmp.shape[0] / defmap.shape[0]) + (tmp.shape[0] % defmap.shape[0] > i) )
            idx = numpy.argsort(tmp[:, 0])
            bins = numpy.zeros(defmap.shape[0]+1)
            k = 0
            for i in xrange(defmap.shape[0]):
                bins[i] = tmp[idx[k], 0]
                k += counts[i]
            bins[defmap.shape[0]] = tmp[idx[idx.shape[0]-1], 0]
        else:
            counts, bins = numpy.histogram(tmp[:, 0], defmap.shape[0])
        for i in xrange(len(counts)):
            print "%f-%f = %d"%(bins[i], bins[i+1], counts[i])
        for i in xrange(defmap.shape[0]):
            sel = numpy.logical_and(bins[i] < tmp[:, 0], tmp[:, 0] <= bins[i+1])
            defmap[i, :] = numpy.sum(tmp[sel, :], axis=0)
            if 1 == 1:
                defmap[i, 0] /= numpy.sum(sel)
            else: defmap[i, 0] = bins[i]
            defmap[i, 0] /= 10000
    print "Defocus:", defmap.mean()
    data = defmap[:, 3] / defmap[:, 1]
    print 'Pre:', data.mean()
    data[defmap[:, 1]==0] = 0
    data = 1.0 - data
    plot_bar(output, defmap[:, 0], data, 'defocus_pre.png', 'False Discovery Rate', 'Defocus ($\mu m$)')
    data = defmap[:, 3] / defmap[:, 2]
    print 'Recall:', data.mean()
    data[defmap[:, 2]==0] = 0
    data = 1.0 - data
    plot_bar(output, defmap[:, 0], data, 'defocus_recall.png', 'False Negative Rate', 'Defocus ($\mu m$)')
    data = defmap[:, 3]
    plot_bar(output, defmap[:, 0], data, 'defocus_overlap.png', 'Overlap', 'Defocus ($\mu m$)')
    
    