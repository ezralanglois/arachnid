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
from arachnid.core.metadata import format, spider_utility
bench;
import numpy #, logging

#format.mrccoord._logger.setLevel(logging.DEBUG)

if __name__ == '__main__':

    # Parameters
    
    #10442 13661
    select = sys.argv[1]
    coords1_file = sys.argv[2]        # image_file = "data/dala_01.spi"
    coords2_file = sys.argv[3]        # align_file = "data/align_01.spi"
    #output_file=sys.argv[4]         # output_file="stack01.spi"
    pixel_radius = 100
    
    #logging.basicConfig(level=logging.DEBUG)
    #ara_max = (0, 0)
    #ben_max = (0, 0)
    for s in format.read(select, numeric=True):
        coords1 = format.read(coords1_file, spiderid=s.araSpiderID, numeric=True)
        if not os.path.exists(spider_utility.spider_filename(coords2_file, s.araLeginonFilename)):
            print s.araLeginonFilename, s.araSpiderID, 0, len(coords1), 0, 0
            continue
        try:
            coords2 = format.read(coords2_file, spiderid=s.araLeginonFilename, numeric=True)
        except:
            print spider_utility.spider_filename(coords2_file, s.araLeginonFilename)
            raise
        
        benchmark = numpy.vstack(([c.x for c in coords2], [c.y for c in coords2])).T.astype(numpy.float)
        autop = numpy.vstack(([c.x for c in coords1], [c.y for c in coords1])).T.astype(numpy.float)
        #ara_max = ( max(ara_max[0], autop[:, 0].max()), max(ara_max[1], autop[:, 1].max()) )
        #ben_max = ( max(ben_max[0], benchmark[:, 0].max()), max(ben_max[1], benchmark[:, 1].max()) )
        overlap = bench.find_overlap(autop, benchmark, pixel_radius)
        #            TP            FP                      TN            FN
        tab = ( len(overlap), len(autop)-len(overlap),    0,  len(benchmark)-len(overlap) )
        print s.araLeginonFilename, s.araSpiderID, " ".join([str(t) for t in tab])
    
    #print 10442, 13661, '->', ara_max
    #print 10442, 13661, '->', ben_max
    
    