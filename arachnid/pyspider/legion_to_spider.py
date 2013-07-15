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
import os, logging, numpy

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
    
    if extra['recover'] != "":
        recover_ids(files, output, **extra)
    else:
        convert_to_spider(files, output)
    
def recover_ids(files, output, reference, recover, **extra):
    '''
    '''
    
    base = os.path.splitext(output)[0]
    output = base+os.path.splitext(files[0])[1]
    select = format_utility.add_prefix(base+".star", 'sel_')
    if not os.path.exists(os.path.dirname(output)): os.makedirs(os.path.dirname(output))
    
    mapping = []
    
    cur_defocus = format_utility.map_object_list(format.read(recover, numeric=True))
    if 1 == 0:
        rdefocus = format.read(reference, numeric=True)
        ref_defocus = {}
        skip_defocus=set()
        for val in rdefocus:
            if val.defocus in ref_defocus:
                del ref_defocus[val.defocus]
                skip_defocus.add(val.defocus)
            elif val.defocus not in skip_defocus:
                ref_defocus[val.defocus]=val.id
    else:
        #cur_defocus, cur_header = format_utility.tuple2numpy(format.read(recover, numeric=True))
        ref_defocus, ref_header = format_utility.tuple2numpy(format.read(reference, numeric=True))
        #astig_ang
        id_idx = ref_header.index('id')
        if 1 == 1:
            ref = ref_defocus[:, (ref_header.index('defocus'), ref_header.index('astig_ang'))]
            idx = numpy.argsort(ref[:, 0], axis=0).squeeze()
        else:
            ref = ref_defocus[:, (ref_header.index('defocus'), ref_header.index('astig_ang'))].view('f4,f4')
            idx = numpy.argsort(ref, axis=0, order=['f0', 'f1']).squeeze()
            ref = ref_defocus[:, (ref_header.index('defocus'), ref_header.index('astig_ang'))]
        
    sref=ref[idx].copy()
    _logger.info("Number that can be mapped: %d"%len(cur_defocus))
    not_in_curr_defocus=0
    ambiguous_count=0
    for filename in files:
        original = os.readlink(filename) if os.path.islink(filename) else filename
        id = spider_utility.spider_id(filename)
        try:
            defocus = cur_defocus[id].defocus
        except:
            not_in_curr_defocus += 1
            #_logger.warn("Skipping: %s - not found in current defocus"%str(id))
            continue
        
        _logger.error("here0: %s -- %s"%(str(defocus), str(sref.shape)))
        _logger.error("here1: %s -- %f"%(str(defocus), float(sref[0, 0])))
        off = numpy.searchsorted(sref[:, 0], float(defocus), 'left')
        _logger.error("here2")
        end = numpy.searchsorted(sref[:, 0], defocus+50, 'left')
        _logger.error("here3")
        if off < end:
            astig_ang = cur_defocus[id].astig_ang
            _logger.error("here4")
            off = numpy.argmin(numpy.abs(sref[off:end, 1]-astig_ang))+off
            #off = numpy.searchsorted(sref[off:end, 1], astig_ang)+off
            _logger.error("here5")
        
        _logger.error("here6: %f == %f"%(defocus, sref[off, 0]))
        if off < ref.shape[0] and numpy.abs(defocus-sref[off, 0]) < 50:
            _logger.error("here7")
            index = int(ref_defocus[idx[off], id_idx])
            _logger.error("here8")
        else:
            _logger.error("here9")
            ambiguous_count += 1
            val = sref[off, 0] if off < len(idx) else -1
            _logger.error("here10")
            _logger.warn("Skipping: %s - with defocus: %f -- %f"%(str(original), defocus, val))
            continue
        
        if 1 == 0:
            try:
                index = ref_defocus[defocus]
            except:
                ambiguous_count += 1
                #_logger.warn("Skipping: %s - with defocus: %f"%(str(original), defocus))
                continue
        _logger.info("%d -> %d for %f"%(id, index, defocus))
        mapping.append((original, index))
        output_file = spider_utility.spider_filename(output, index)
        if os.path.exists(output_file): os.unlink(output_file)
        os.symlink(os.path.abspath(original), output_file)
    _logger.info("Skipped - Defocus: %d - Ambiguous: %d"%(not_in_curr_defocus, ambiguous_count))
    
    
    format.write(select, mapping, header="araLeginonFilename,araSpiderID".split(','))

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
    if not os.path.exists(os.path.dirname(output)): os.makedirs(os.path.dirname(output))
    mapping = [v for v in format.read(select, numeric=True) if v.araSpiderID > 0 and os.path.exists(v.araLeginonFilename)] if os.path.exists(select) else []
    mapped = dict([(v.araLeginonFilename, v.araSpiderID) for v in mapping if v.araSpiderID > 0 and os.path.exists(v.araLeginonFilename)])
    update = [f for f in files if f not in mapped]
    index = len(mapped)+offset
    for f in update:
        output_file = spider_utility.spider_filename(output, index+1)
        if os.path.exists(output_file): os.unlink(output_file)
        try:
            os.symlink(os.path.abspath(f), output_file)
        except:
            os.unlink(output_file)
            try:
                os.symlink(os.path.abspath(f), output_file)
            except:
                _logger.error("%s"%output_file)
                raise
        mapping.append((f, index+1))
        index += 1
    format.write(select, [tuple(m) for m in mapping], header="araLeginonFilename,araSpiderID".split(','))
    return [spider_utility.spider_filename(output, v) for v in xrange(1, len(mapping)+1) if os.path.exists(spider_utility.spider_filename(output, v))]

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
    base = None
    for f in files:
        if base is None:
            base = spider_utility.spider_basename(f)
        elif base != spider_utility.spider_basename(f):
            return True
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
    pgroup.add_option("-r", reference="", help="Recovery reference defocus file or power spectra spider template - has the proper IDs")
    pgroup.add_option("-y", recover="", help="Recovery current defocus file or power spectra spider template")
    
def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if not is_legion_filename(options.input_files) and options.recover == "":
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


