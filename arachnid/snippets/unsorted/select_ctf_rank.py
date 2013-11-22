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
#import sys
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1/')
from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file

if __name__ == '__main__':

    # Parameters
    
    selection_file = ""
    defocus_file = ""
    rank_cutoff = 0.6
    output_file=""
    stack_file=""
    
    selection, header = format.read(selection_file, ndarray=True)
    rank = header.index('rank')
    id = header.index('id')
    def_col = header.index('defocus')
    # 1.0000       33650.      -28.196       1301.7      0.15722
    defocus_map = format_utility.map_object_list(format.read(defocus_file, numeric=True, header="id,defocus,astig_mag,astig_ang,cutoff".split(','))) 
    selection = selection[selection[:, rank]>rank_cutoff]
    total = 0
    for s in selection:
        val = defocus_map.get(int(s[id]), None)
        if val is not None: s[def_col] = val.defocus
        if stack_file != "": total += ndimage_file.count_images(spider_utility.spider_filename(stack_file, int(s[id])))
    print "Total projects: ", total
    format.write(output_file, selection, default_format=format.spiderdoc, header=header)