''' Merge the output from montagefromdoc and ctf-select

Download to edit and run: :download:`merge_select.py <../../arachnid/snippets/merge_select.py>`

To run:

.. sourcecode:: sh
    
    $ python merge_select.py

.. literalinclude:: ../../arachnid/snippets/merge_select.py
   :language: python
   :lines: 16-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility


if __name__ == '__main__':

    # Parameters
    
    montage_select_file = ""
    ctf_select_file = ""
    output_file = ""
    
    micrographs = format_utility.map_object_list(format.read(ctf_select_file, numeric=True))
    selection = format.read(montage_select_file, numeric=True, header='id,select'.split(','))
    subset=[]
    for sel in selection:
        try:
            subset.append(micrographs[sel.id])
        except:
            print "Id not found", sel.id
    format.write(output_file, subset, default_format=format.spiderdoc)
