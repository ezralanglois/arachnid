''' Post process the individual micrograph selection files written by Orbweaver

Download to edit and run: :download:`postprocess_mic_select.py <../../arachnid/snippets/postprocess_mic_select.py>`

To run:

.. sourcecode:: sh
    
    $ python postprocess_mic_select.py

.. literalinclude:: ../../arachnid/snippets/postprocess_mic_select.py
   :language: python
   :lines: 16-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility
import glob


if __name__ == '__main__':

    # Parameters
    
    input_file = "sel*.ext"
    output_file = ""
    
    input_files = glob.glob(input_file)
    vals = []
    SelectTuple = format_utility.namedtuple("Selection", "id,select")
    for filename in input_files:
        vals.extend([SelectTuple(val.oid, val.select) for val in format.read(filename, numeric=True) if val.select > 0])
    format.write(output_file, vals, default_format=format.spiderdoc)


