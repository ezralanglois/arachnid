''' Test if files contain enumeration, if not generate a set of enumerating soft links

This script tests if a list of filenames contains numbers in the power format 
and it not, creates a set of soft links with the proper naming convention: file_00001.ext

Notes
=====

    #. An output selection file is created that maps the orignal name to
       the enumerated name.
    #. The `--mapping-file` option allows the user to relink the target
       files after they have moved to another directory.

Examples
========

.. sourcecode :: sh

    $ ara-enumfiles path1/*en.mrc -o path/file_000001.dat
    
    $ ara-enumfiles path2/*en.mrc -o new-path/file_000001.dat -m orig/sel_file.dat


Critical Options
================

.. program:: ara-enumfiles

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames
    If you use the parameters `-i` or `--inputfiles` the filenames may be comma or 
    space separated on the command line; they must be comma seperated in a configuration 
    file. Note, these flags are optional for input files; the filenames must be separated 
    by spaces. For a very large number of files (>5000) use `-i "filename*"`

.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output template for the enumerated filename (e.g. mic_0000.mrc)

Useful Options
===============

.. program:: ara-enumfiles

.. option:: -m, --mapping-file <FILENAME>

    Input filename for selection file that maps the original 
    filename to the enumerated filename. This is specified when the files need to be
    relinked because of a change in location.

.. option:: --ignore-duplicate <BOOL>
    
    Ignore duplicates found with same filename but in a different path location

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`

.. Created on Nov 23, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program, tracing
from ..core.metadata import format_utility, spider_utility, format
import os
import logging


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, mapping_file="", **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
    
    files : list
            List of input filenames
    output : str
             Output filename for micrograph links
    mapping_file : str
                   Filename of possible existing mapping
    extra : dict
            Unused keyword arguments
    '''
    
    if is_enum_filename(files):
        _logger.info("All files follow the proper convention!")
        return
    files = sorted(files)    
    output_map_file = format_utility.add_prefix(os.path.splitext(spider_utility.spider_filepath(output))[0]+".star", 'sel_')
    if mapping_file == "":
        mapped = map_enum_files(files, output_map_file)
    else:
        mapped = remap_enum_files(files, mapping_file)
    output = os.path.splitext(output)[0]+os.path.splitext(files[0])[1]
    format.write(output_map_file, mapped, header="filename,id".split(','))
    generate_enum_links(mapped, output)
    _logger.info("Completed")
    
def map_enum_files(files, mapping_file):
    ''' Map the list of files, optionally, given an existing mapping
    
    :Parameters:
    
    files : list
            List of filenames to map
    mapping_file : str
                   Filename of possible existing mapping
    
    :Returns:
    
    mapped : list
             List of mapped files
    '''
    
    if os.path.exists(mapping_file):
        mapped = format.read(mapping_file, numeric=True)
        tracing.backup(mapping_file)
    else:
        mapped = []
    mapset = set([v.filename for v in mapped])
    basemap = dict([(os.path.basename(v.filename), i) for i, v in enumerate(mapped)])
    mapped = [tuple(m) for m in mapped]
    index = len(mapped)+1
    relink = 0
    for filename in files:
        if filename not in mapset:
            index = basemap.get(os.path.basename(filename), None)
            if index is not None:
                mapped[index] = (filename, mapped[index][1])
                relink += 1
            else:
                mapped.append((filename, index))
                index += 1
    if relink > 0:
        _logger.info("Relinked %d of %d files"%(relink, len(files)))
    return mapped
    
def remap_enum_files(files, mapping_file):
    ''' Remap the list of files given an existing mapping
    
    :Parameters:
    
    files : list
            List of filenames to remap
    mapping_file : str
                   Filename of existing mapping
    
    :Returns:
    
    mapped : list
             List of tuples mapping filename to id
    '''
    
    mapped = format.read(mapping_file, numeric=True)
    mapped = [tuple(m) for m in mapped]
    newmapping=[]
    index = len(mapped)+1
    for filename in files:
        basename = os.path.basename(filename)
        found=[]
        for pair in mapped:
            if pair[0].find(basename) != -1 or basename.find(pair[0]) != -1:
                found.append(pair)
        if len(found) == 1:
            pair = found[0]
            newmapping.append((filename, pair[1]))
        elif len(found) > 1:
            _logger.error('Duplicate mappings (%d) found for filename %s -> %s'%(len(found), filename, str(found)))
            raise ValueError, 'Duplicate mappings (%d) found for filename %s -> %s'%(len(found), filename, str(found))
        else:
            newmapping.append((filename, index))
            index += 1
            #_logger.error('Filename not found in mapping: %s'%filename)
            #raise ValueError, 'Filename not found in mapping: %s'%filename
    if len(newmapping) < len(mapped):
        _logger.warn("New mapping smaller than original: %d < %d"%(len(newmapping), len(mapped)))
    if len(newmapping) > len(mapped):
        _logger.warn("New mapping larger than original: %d > %d"%(len(newmapping), len(mapped)))
    return newmapping

def generate_enum_links(mapped, output):
    ''' Generate a set of enumerated softlinks
    
    :Parameters:
    
    mapped : list
             List of tuples mapping filename to id
    output : str
             Output link filename template
    '''
    
    if not os.path.exists(os.path.dirname(output)):
        try: os.makedirs(os.path.dirname(output))
        except: pass
    for filename, id in mapped:
        link = spider_utility.spider_filename(output, id)
        if os.path.exists(link): os.unlink(link)
        if os.path.exists(filename):
            os.symlink(os.path.abspath(filename), link)
        else:
            _logger.warn("Unlinking %s because %s does not exist"%(link, filename))

def is_enum_filename(files):
    ''' Test if list of filenames all follow the enumerated filename
    convention.
    
    An enumerated filename starts with some prefix, followed by a number
    and then the file extension: path/filename00001.dat
    
    :Parameters:
    
    files : list
            List of filenames to test
    
    :Returns:
    
    flag : bool
           True if a single filename does not follow this convention or
           if the ID is not unique
    '''
    
    found = set()
    for f in files:
        try:
            id = spider_utility.spider_id(f)
        except: 
            _logger.debug("No id: "+str(f))
            return False
        else:
            if id in found: 
                _logger.debug("Found id: "+str(id)+" in "+str(found))
                return False
            found.add(id)
    return True

def supports(files, **extra):
    ''' Test if this module is required in the project workflow
    
    :Parameters:
    
    files : list
            List of filenames to test
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    flag : bool
           True if this module should be added to the workflow
    '''
    
    return not is_enum_filename(files)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if len(options.input_files)==0: raise OptionValueError, "No input filenames"
    if not spider_utility.is_spider_filename(options.output): raise OptionValueError, "Output filename not a enumerated filename, e.g. path/mic_00000.spi"

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
        
    pgroup.add_option("-i", "--unenum-files", input_files=[],         help="List of input filenames containing micrographs", required_file=True, gui=dict(filetype="open"))
    pgroup.add_option("-o", "--linked-files", output="",              help="Output filename for micrograph links", gui=dict(filetype="save"), required=True)
    pgroup.add_option("-m", mapping_file="",        help="Recreate mapping after files have changed in location", gui=dict(filetype="open"))

def flags():
    ''' Get flags the define the supported features
    
    :Returns:
    
    flags : dict
            Supported features
    '''
    
    return dict(description = '''Generate a set of enumerated soft links to files that do not conform to the Arachnid filename convention
                                 
                                 $ %prog *en.mrc -o path/file_000001.dat
                              ''',
                supports_MPI=False, 
                supports_OMP=False,
                use_version=False)

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__)

if __name__ == "__main__": main()


