''' Select overlapping tilt pair coordinates and write to new file

Download to edit and run: :download:`tilt_coord_register.py <../../arachnid/snippets/tilt_coord_register.py>`

To run:

.. sourcecode:: sh
    
    $ python tilt_coord_register.py

.. literalinclude:: ../../arachnid/snippets/tilt_coord_register.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1/')
from arachnid.core.metadata import format, format_utility

if __name__ == '__main__':

    # Parameters
    
    selection_file = sys.argv[1]
    coord_file = sys.argv[2]
    output_file = sys.argv[3]
    
    selection = format.read(selection_file, numeric=True)
    selection = format_utility.map_file_list(selection, 'mic1')
    
    for values in selection.itervalues():
        coords = format.read(coord_file, numeric=True, spiderid=values[0].mic1)
        coords = [coords[v.id-1] for v in values]
        format.write(output_file, coords, spiderid=values[0].mic1)
        coords = format.read(coord_file, numeric=True, spiderid=values[0].mic2)
        coords = [coords[v.id2-1] for v in values]
        format.write(output_file, coords, spiderid=values[0].mic2)
    
    
    