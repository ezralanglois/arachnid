''' Combine and invert selection file

Download to edit and run: :download:`invert_selection.py <../../arachnid/snippets/invert_selection.py>`

To run:

.. sourcecode:: sh
    
    $ python invert_selection.py

.. literalinclude:: ../../arachnid/snippets/invert_selection.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys, glob
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1/')
from arachnid.core.metadata import format, spider_utility

if __name__ == '__main__':

    # Parameters
    
    win_files = sys.argv[1]
    sel_files = sys.argv[2]
    output_file=sys.argv[3]
    
    win_files = glob.glob(win_files)
    sel_files = glob.glob(sel_files)
    
    fullsel = {}
    for i in xrange(len(win_files)): fullsel[spider_utility.spider_id(win_files[i])] = 1
    for sel_file in sel_files:
        sel = format.read(sel_file, numeric=True)
        for s in sel: fullsel[s.id]=0
    format.write(output_file, fullsel.items(), header='id,select'.split(','), default_format=format.spidersel)
        