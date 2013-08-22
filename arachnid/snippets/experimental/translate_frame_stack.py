''' This script creates stacks from translation corrected frames

Download to edit and run: :download:`translate_frame_stack.py <../../arachnid/snippets/translate_frame_stack.py>`

To run:

.. sourcecode:: sh
    
    $ python translate_frame_stack.py

.. literalinclude:: ../../arachnid/snippets/translate_frame_stack.py
   :language: python
   :lines: 17-
   :linenos:
'''
import sys

#from arachnid.core.metadata import format, spider_utility

if __name__ == '__main__':

    # Parameters
    image_selfile = sys.argv[1]
    trans_selfile = sys.argv[2]
    coords_selfile = sys.argv[3]
    output_file = sys.argv[4]
    
    