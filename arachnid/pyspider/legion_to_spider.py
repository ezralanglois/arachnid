''' Creates a set of soft-links with SPIDER compatible filenames

This |spi| batch file generates a set of soft links to map micrographs
captured on Leginon to SPIDER compatible names.

Examples
========

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Create a project directory and scripts using 4 cores
    
    $ spi-leginon2spi mic_*.tif
    
Critical Options
================

.. program:: spi-project

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames containing micrographs.
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`

.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for micrograph links (Default: mapped_micrographs/mic_0000000)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`

.. Created on Nov 2, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.metadata import spider_utility, format, format_utility
import os, logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
    
    files : list
            List of input filenames
    output : str
             Output filename for micrograph links
    extra : dict
            Unused keyword arguments
    '''
    
    convert_to_spider(files, output)

def convert_to_spider(files, output, offset=0):
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
    select = format_utility.add_prefix(base+".star", 'sel_')
    output = base+os.path.splitext(files[0])[1]
    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))
    mapping = format.read(select, numeric=True) if os.path.exists(select) else []
    mapped = dict([(v.araLeginonFilename, v.araSpiderID) for v in mapping])
    update = [f for f in files if f not in mapped]
    index = len(mapped)+offset
    for f in update:
        output_file = spider_utility.spider_filename(output, index+1)
        if os.path.exists(output_file): os.unlink(output_file)
        os.symlink(os.path.abspath(f), output_file)
        mapping.append((f, index+1))
        index += 1
    format.write(select, mapping, header="araLeginonFilename,araSpiderID".split(','))
    return [spider_utility.spider_filename(output, v) for v in xrange(1, len(mapping)+1)]

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

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
        
    pgroup.add_option("-i", input_files=[],     help="List of input filenames containing micrographs", required_file=True, gui=dict(filetype="file-list"))
    pgroup.add_option("-o", output="mapped_micrographs/mic_0000000", help="Output filename for micrograph links", gui=dict(filetype="save"), required=True)
    
def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if not is_legion_filename(options.input_files):
        raise OptionValueError, "Filenames appear to be in SPIDER format already!"

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Create a set of soft links to map Leginon to SPIDER compatible filenames
                        
                        $ %prog micrograph_files* -o path/mic_00000
                      ''',
        supports_MPI=False,
        use_version = False,
    )
if __name__ == "__main__": main()


