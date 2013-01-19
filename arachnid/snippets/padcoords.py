''' This script adds a select column to a dataset

Download to edit and run: :download:`padcoords.py <../../arachnid/snippets/padcoords.py>`

To run: python padcoords.py "../data/coords/sndc*.ter" x,y,id,peak coords/sndc_15787.ter win/win_15787.ter

.. sourcecode:: sh
    
    $ python padcoords.py

.. literalinclude:: ../../arachnid/snippets/padcoords.py
   :language: python
   :lines: 17-
   :linenos:
'''
import sys, os
sys.path.append(os.path.expanduser('~/workspace/arachnida/src'))
from arachnid.core.metadata import format, spider_utility
from arachnid.core.image import ndimage_file
import numpy, glob

if __name__ == '__main__':

    # Parameters
    
    input_file = sys.argv[1]
    header = sys.argv[2]
    good_file = sys.argv[3]
    match_file = sys.argv[4]
    pad_file = sys.argv[5]
    output_file = sys.argv[6]
    
    count = 0
    off = 0
    for filename in glob.glob(input_file):
        if not os.path.exists(spider_utility.spider_filename(match_file, filename)): continue
        full = format.read(filename, numeric=True, header=header.split(','))
        selected = numpy.asarray(format.read(good_file, spiderid=filename, numeric=True, header="id,select".split(',')))[:, 0]-1
        coords = []
        for s in selected: coords.append(full[s])
        curr = ndimage_file.count_images(spider_utility.spider_filename(match_file, filename))
        prev = len(coords)
        total = curr-len(coords)
        if total > 0:
            coords.extend(format.read(pad_file, spiderid=filename, numeric=True)[:total])
        count += len(coords)
        off += 1
        print off, count, prev, total, curr, len(coords)
        format.write(output_file, coords, spiderid=filename, default_format=format.spiderdoc)


