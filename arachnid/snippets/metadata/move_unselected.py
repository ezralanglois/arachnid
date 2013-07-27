''' Move unselected files to a directory

.. Created on Jul 22, 2013
.. codeauthor:: robertlanglois
'''
from arachnid.core.metadata import format, spider_utility
import os, sys

if __name__ == '__main__':

    # Parameters
    
    select_file = sys.argv[1]
    map_file = sys.argv[2]
    output_dir = sys.argv[3]
    
    name_map = format.read(map_file)
    
    selected = {}
    for select in format.read(select_file, numeric=True):
        selected[select.id]=select.select
    
    for val in name_map:
        if selected.get(spider_utility.spider_id(val.araSpiderID), 0) > 0: continue
        input_stack = os.path.basename(val.araLeginonFilename)
        n = input_stack.find('_')
        input_stack=input_stack[n+1:]
        try:
            os.rename(input_stack, os.path.join(output_dir, input_stack))
        except:
            print input_stack, '->', os.path.join(output_dir, input_stack)
            raise

        
        



