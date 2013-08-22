''' This script relinks frame average images to the frame stacks

Download to edit and run: :download:`relink_frames.py <../../arachnid/snippets/relink_frames.py>`

To run:

.. sourcecode:: sh
    
    $ python relink_frames.py

.. literalinclude:: ../../arachnid/snippets/relink_frames.py
   :language: python
   :lines: 17-
   :linenos:
'''
import sys,os

from arachnid.core.metadata import format, spider_utility

if __name__ == '__main__':

    # Parameters
    selfile = sys.argv[1]
    trans_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    vals = format.read(selfile)
    for v in vals:
        basename = "gc_"+os.path.splitext(os.path.basename(v.araLeginonFilename)[6:])[0]+"_Log.txt.csv"
        try:
            os.rename(os.path.join(trans_dir, basename), spider_utility.spider_filename(output_dir, v.araSpiderID))
        except: pass


