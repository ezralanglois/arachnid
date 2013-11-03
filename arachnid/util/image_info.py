''' Retrieve information from the header of an image

This script (`ara-info`) retrieves metadata from the header of an image and
prints this information to the console (STDOUT). By default, it prints 
the most pertient information: image dimensions, number of images and pixel
spacing (see example below).

Examples
========
    
.. sourcecode:: sh

 $ ara-info mics/al_TSl12_000001.dat mics/al_TSl12_000002.dat
    name                                                nx  ny  nz  count   apix   
    al_TSl12_000001.dat                                3710 3838   1       0    0.00
    al_TSl12_000002.dat                                3710 3838   1       0    0.00

Critical Options
================

.. program:: ara-info

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of filenames for the input images.
    |input_files|

Useful Options
===============

These options 

.. program:: ara-info

.. option:: -a, --all
    
    Show all the information contained in the header. Note
    the output format changes.
    
.. option:: -s, --stat
    
    Calculate and displays simple statistics for each image,
    which include: mean, standard deviation, max, min and
    number of unique values.
    
.. option:: -f, --force
    
    Use EMAN2/Sparx formats (if available) instead of internal 
    image formats: SPIDER and MRC

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`

.. Created on Apr 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.image import ndimage_file
import logging, os, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output="", all=False, stat=False, force=False, **extra):
    ''' Retrieve information from the header of an image
    
    :Parameters:
    
    filename : tuple
               File index and input filename
    all : bool
          Print all header information
    stat : bool
           Calculate statistics of the image
    force : bool
            If EMAN2 is available, use its image formats (override internal spider and mrc formats)
    extra : dict
            Unused key word arguments
    '''
    
    size = (None, None, None)
    for i, filename in enumerate(files):
        if force and ndimage_file.eman_format.is_avaliable() and ndimage_file.eman_format.is_readable(filename):
            header = ndimage_file.eman_format.read_header(filename)
        else:
            header = ndimage_file.read_header(filename)
        header['name'] = os.path.basename(filename)
        if all:
            #count = ndimage_file.count_images(filename)
            for key, val in header.iteritems():
                print key, ": ", val
            if stat:
                for img in ndimage_file.iter_images(filename):
                    print "Mean: ", numpy.mean(img)
                    print "STD: ", numpy.std(img)
                    print "Max: ", numpy.max(img)
                    print "Min: ", numpy.min(img)
                    print "Unique: ", len(numpy.unique(img))
        else:
            if i == 0:
                header_name = dict([(k, k) for k in header.iterkeys()])
                print '{name:50}  {nx:3} {ny:3} {nz:3} {count:7} {apix:7}'.format(**header_name)
            print '{name:50} {nx:3d} {ny:3d} {nz:3d} {count:7d} {apix:7.2f}'.format(**header)
       
    _logger.info("Complete")

def setup_options(parser, pgroup=None, main_option=False):
    ''' Add options to OptionParser for application
    
    :Parameters:
    
    parser : OptionParser
             Object defining an OptionParser class
    pgroup : OptionGroup
             Options specific to running the script
    main_option : bool
                  If true, then add options specific to running the script
    '''
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Image information", "Retrieve information from the header of an image",  id=__name__)
    group.add_option("-a", all=False,                     help="Show all the information contained in the header. Note the output format changes.")
    group.add_option("-s", stat=False,                    help="Calculate and displays simple statistics for each image, which include: mean, standard deviation, max, min and number of unique values.")
    if ndimage_file.eman_format.is_avaliable():
        group.add_option("-f", force=False,               help="Use EMAN2/Sparx formats (if available) instead of internal image formats: SPIDER and MRC")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the images", required_file=True, gui=dict(filetype="file-list"))
        #pgroup.add_option("-o", output="",      help="Output filename for the relion selection file", gui=dict(filetype="save"), required_file=True)
        parser.change_default(log_level=logging.WARN)

def main():
    ''' Main entry point for the script
    '''
    
    run_hybrid_program(__name__,
        description = ''' Retrieve information from the header of an image
                         
                         Example:
                         
                         $ ara-info image.spi
                      ''',
        supports_MPI = False,
        use_version = False,
    )
def dependents(): return []
if __name__ == "__main__": main()



