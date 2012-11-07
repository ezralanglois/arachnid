''' Benchmark particle selection

This script (`ara-bench`) is designed to benchmark particle selection against another
program or human.

Notes
=====

+---------------------+---------------------------------+-------------------+
|                     |           Actual Class          | Total             |
+=====================+================+================+===================+
|                     | True Positive  | False Positive | Total Predicted   |
|                     | (TP)           | (FP)           | Positive (P')     |
| **Predicted Class** +----------------+----------------+-------------------+
|                     | False Negative | True Negative  | Total Predicted   |
|                     | (FN)           | (TN)           | Negative (N')     |
+---------------------+----------------+----------------+-------------------+
| **Total**           | Total Actual   | Total Actual   |                   |
|                     | Positive (P)   | Negative (N)   |                   |
+---------------------+----------------+----------------+-------------------+

The following metrics are supported:

=========== ============= ===========================================================
Name        Equation      Description
=========== ============= ===========================================================
Precision   |precision|   The fraction of positives correctly predicted positive
Recall      |recall|      The fraction of positives discovered from the total
=========== ============= ===========================================================


.. Metric Definitions
.. ------------------
.. |precision| replace:: :math:`TP \over{FP+TP}`
.. |recall| replace:: :math:`TP\over{TP+FN}`
.. |roc_curve| replace:: :math:`\left( \\frac{TN}{TN+FP}, \\frac{TP}{FP+TP}\\right)`
.. |prc_curve| replace:: :math:`\left( \\frac{TP}{TP+FN}, \\frac{TP}{FP+TP}\\right)`

Examples
========

.. sourcecode :: sh

    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Run with a disk as a template on a raw film micrograph
    
    $ ara-bench sndc_*.spi -o sndc_00001.spi -r 110 -w 312
    
Critical Options
================

.. program:: ara-bench

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of filenames for the input coordinate file.
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`

.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for the confusion matrix

.. option:: -r <int>, --pixel-radius <int>
    
    Size of your particle in pixels (used to determine the amount of allowed overlap).
    
.. option:: -g <FILENAME>, --good <FILENAME>
    
    Selection file listing good particles for performance benchmark

.. option:: --good-coords <FILENAME>
    
    Coordinates for the good particles for performance benchmark (if different from the input coordinates)

Useful Options
===============

.. program:: ara-bench

.. option:: --good-output <FILENAME>
    
    Output coordinates for the good particles for performance benchmark

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`

.. todo:: add by defocus

.. Created on Sep 24, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app.program import run_hybrid_program
from ..core.metadata import format_utility, format, spider_utility
from ..core.parallel import mpi_utility
import os, logging
import numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, **extra):
    '''Concatenate files and write to a single output file
    
    :Parameters:
        
    filename : str
               Input filename
    output : str
             Filename for output file
    extra : dict
            Unused key word arguments
                
    :Returns:
        
    filename : string
          Current filename
    peaks : numpy.ndarray
             List of peaks and coordinates
    '''
    
    coords = format.read(filename, numeric=True)
    confusion = benchmark(coords, filename, **extra)
    return filename, confusion

def benchmark(coords, fid, good_output="", **extra):
    ''' Benchmark the specified set of coordinates against a gold standard
    
    :Parameters:
    
    coords : array
             List of x,y coordinate centers for particles
    fid : str
          Current SPIDER id
    good_output : str
                  Output filename for good particle coordinates
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    conf : tuple
           Confusion matrix counts: (TP, FP, TN, FN)
    '''
    
    coords, header = format_utility.tuple2numpy(coords)
    bench = read_bench_coordinates(fid, **extra)
    if bench is not None:
        selected = bench
        x, y = header.index('x'), header.index('y')
        overlap = find_overlap(numpy.vstack((coords[:, x], coords[:, y])).T, bench, **extra)
    else:
        selected = format_utility.tuple2numpy(format.read(extra['good'], spiderid=fid, id_len=extra['id_len'], numeric=True))[0].astype(numpy.int)
        overlap = coords[selected].copy().squeeze()
    if good_output != "":
        format.write(spider_utility.spider_filename(good_output, fid), overlap, header=header)
           #    TP            FP                        TN         FN
    return ( len(overlap), len(coords)-len(overlap),    0,  len(selected)-len(overlap) )
    
def read_bench_coordinates(fid, good_coords="", good="", id_len=0, **extra):
    ''' Read benchmark coordinates
    
    :Parameters:
        
    pixel_radius : int
                   Current SPIDER ID
    good_coords : str
                  Filename for input benchmark coordinates
    good : str
           Filename for optional selection file to select good coordinates
    id_len : int
             Maximum length of the ID
    extra : dict
            Unused keyword arguments
    
    :Returns:
        
    coords : numpy.ndarray
             Selected (x,y) coordinates
    '''
    
    if good_coords == "": return None
    if not os.path.exists(spider_utility.spider_filename(good_coords, fid, id_len)):
        coords, header = format_utility.tuple2numpy(format.read(good_coords, numeric=True))
    else:
        coords, header = format_utility.tuple2numpy(format.read(good_coords, spiderid=fid, id_len=id_len, numeric=True))
    if good != "":
        try:
            selected = format_utility.tuple2numpy(format.read(good, spiderid=fid, id_len=id_len, numeric=True))[0].astype(numpy.int)
        except:
            return None
        else:
            selected = selected[:, 0]-1
            coords = coords[selected].copy().squeeze()
    x, y = header.index('x'), header.index('y')
    return numpy.vstack((coords[:, x], coords[:, y])).T

def find_overlap(coords, benchmark, pixel_radius, bench_mult=1.2, **extra):
    '''Find the overlaping coordinates with the benchmark
    
    :Parameters:
        
    coords : numpy.ndarray
             Coorindates found by algorithm
    benchmark : numpy.ndarray
                Set of 'good' coordinates
    pixel_radius : int
                   Radius of the particle in pixels
    bench_mult : float
                 Amount of allowed overlap (pixel_radius*bench_mult)
    
    :Returns:
        
    coords : list
             Overlapping (x,y) coordinates
    '''
    
    assert(benchmark.shape[1] == 2)
    benchmark=benchmark.copy()
    rad = pixel_radius*bench_mult
    pixel_radius = rad*rad
    selected = []
    if len(benchmark) > 0:
        for i, f in enumerate(coords):
            dist = f-benchmark
            numpy.square(dist, dist)
            dist = numpy.sum(dist, axis=1)
            if dist.min() < pixel_radius:
                benchmark[dist.argmin(), :]=(1e20, 1e20)
                selected.append((i+1, 1))
    return selected

def precision(tp, fp, tn, fn):
    ''' Estimate the precision from a confusion matrix
    
    |precision|
    
    :Parameters:
    
    tp : float
         True positive rate
    fp : float
         False positive rate
    tn : float
         True negative rate
    fn : float
         False negative rate
    
    :Returns:
    
    precision : float
                Esimated precision
    '''
    
    return numpy.divide(tp, fp+tp)

def recall(tp, fp, tn, fn):
    ''' Estimate the recall from a confusion matrix
    
    |recall|
    
    :Parameters:
    
    tp : float
         True positive rate
    fp : float
         False positive rate
    tn : float
         True negative rate
    fn : float
         False negative rate
    
    :Returns:
    
    precision : float
                Esimated precision
    '''
    
    return numpy.divide(tp, fn+tp)

def initialize(files, param):
    # Initialize global parameters for the script
    
    param["confusion"] = numpy.zeros((len(files), 4))
    
    if mpi_utility.is_root(**param):
        if not os.path.exists(os.path.dirname(param['output'])):
            os.makedirs(os.path.dirname(param['output']))
        _logger.info("Pixel radius: %d"%param['pixel_radius'])

def reduce_all(val, confusion, file_index, **extra):
    # Process each input file in the main thread (for multi-threaded code)

    filename, conf = val
    if numpy.sum(conf) > 0:
        confusion[file_index, :] = conf
        pre = precision(*conf)
        sen = recall(*conf)
        info = " - %d,%d,%d - precision: %f, recall: %f"%(conf[0]+conf[1], conf[0]+conf[3], conf[0], pre, sen)
    else: info=""
    _logger.info("Finished processing: %s%s"%(os.path.basename(filename), info))
    return filename

def finalize(files, confusion, output, **extra):
    # Finalize global parameters for the script
    
    tot = numpy.sum(confusion, axis=0)
    if tot[1] > 0:
        pre = precision(*tot)
        sen = recall(*tot)
        _logger.info("Overall - precision: %f, recall: %f - %d,%d,%d"%(pre, sen, tot[0]+tot[1], tot[0]+tot[3], tot[0]))
        format.write(os.path.splitext(output)[0]+".csv", confusion, header="tp,fp,tn,fn".split(','), prefix="summary")
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    bgroup = OptionGroup(parser, "Benchmarking", "Options to control benchmark particle selection",  id=__name__)
    bgroup.add_option("-g", good="",        help="Selection file listing good particles for performance benchmark", gui=dict(filetype="open"))
    bgroup.add_option("",   good_coords="", help="Coordinates for the good particles for performance benchmark (if different from the input coordinates)", gui=dict(filetype="open"))
    bgroup.add_option("",   good_output="", help="Output coordinates for the good particles for performance benchmark", gui=dict(filetype="open"))
    pgroup.add_option_group(bgroup)
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-r", pixel_radius=0, help="Radius of the expected particle (if default value 0, then overridden by SPIDER params file, --param-file)")

def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    if main_option:
        if options.good == "" and options.good_coords == "":
            raise OptionValueError, "Requires a --good selection file or a --good-coords file or both"
    if options.good_coords != "" and options.pixel_radius == 0: raise OptionValueError, "Pixel radius must be greater than zero"

def main():
    #Main entry point for this script
    run_hybrid_program(__name__,
        description = '''Benchmarking particle selection
        
                        http://
                        
                        Example
                         
                        $ ara-bench sndc_*.dat -o coords.dat -r 110
                      ''',
        supports_MPI = False,
        use_version = False,
    )

if __name__ == "__main__": main()

