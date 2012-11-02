''' Creates a set of soft-links with SPIDER compatible filenames

.. Created on Nov 2, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.metadata import spider_utility
import os, logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def convert_to_spider(files, output):
    ''' Create a folder of soft links to each leginon micrograph and
    a selection file mapping each SPIDER file to the original leginon
    filename
    
    :Parameters:
    
    files : list
            List of micrograph files
    output : str
             Output filename for micrograph links
    
    :Returns:
    
    files : list
            List of mapped files
    '''
    
    if len(files)==0: return files
    base = os.path.splitext(output)[0]
    output = base+os.path.splitext(files[0])[1]
    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))
    newfiles = []
    mapping = []
    for i, f in enumerate(files):
        output_file = spider_utility.spider_filename(output, i+1)
        os.symlink(os.path.abspath(f), output_file)
        mapping.append((f, i+1))
        newfiles.append(output_file)
    format.write(base+".star", mapping, header="araLeginonFilename,araSpiderID".split(','), prefix="sel_")
    return newfiles

def is_legion_filename(files):
    ''' Test if filename is in leginon format
    
    :Parameters:
    
    files : list
            List of filenames to test
    
    :Returns:
    
    test : bool
           True if filename in leginon format
    '''
    
    found = set()
    for f in files:
        try:
            id = spider_utility.spider_id(f)
        except: return True
        else:
            if id in found: return True
            found.add(id)
    return False
